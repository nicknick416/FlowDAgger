"""Run the real ARX FlowDAgger offline pipeline without touching hardware."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from arx_adapter import CAMERA_KEYS, PROTOCOL_VERSION, extract_vlm_feature
from arx_campaign import add_arx_runtime_args, preload_campaign_config
from arx_episode_store import EpisodeStore
from arx_flowdagger_server import ARXFlowDaggerServer
from arx_trainer import ARXFlowDaggerRuntime


def _state() -> np.ndarray:
    state = np.zeros(20, dtype=np.float32)
    state[[0, 10]] = 0.3
    state[[2, 12]] = 0.2
    state[3:9] = state[13:19] = [1, 0, 0, 0, 1, 0]
    state[[9, 19]] = 0.07
    return state


def _images() -> dict[str, np.ndarray]:
    return {
        key: np.zeros((224, 224, 3), dtype=np.uint8) for key in CAMERA_KEYS
    }


def _message_images(images: dict[str, np.ndarray]) -> dict[str, list]:
    return {key: value.transpose(2, 0, 1).tolist() for key, value in images.items()}


def main() -> None:
    cfg = preload_campaign_config()
    parser = argparse.ArgumentParser()
    add_arx_runtime_args(parser, cfg)
    args = parser.parse_args()
    prompt = args.default_prompt
    run_root = Path(args.output_root) / "offline_validation" / (
        "end_to_end_" + time.strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=False)

    runtime = ARXFlowDaggerRuntime(
        openpi_root=args.openpi_root,
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        output_root=str(run_root),
        default_prompt=prompt,
        inversion_batch_size=1,
    )
    state = _state()
    images = _images()
    observation = {**images, "state": state, "prompt": prompt}
    known_noise = np.random.default_rng(7).normal(size=(50, 32)).astype(np.float32)
    expert_actions = np.asarray(
        runtime.policy.infer(observation, noise=known_noise)["actions"],
        dtype=np.float32,
    )

    store = EpisodeStore(run_root)
    episode_dirs = []
    # Two isolated episodes are required by the real offline validation gate.
    # Each contains five independent two-sample segments to exercise padding.
    for episode_index in range(2):
        store.start(
            f"smoke_demo_{episode_index}",
            prompt=prompt,
            base_policy_version=0,
            steering_policy_version=0,
            shadow_mode=True,
            run_stage="demonstration",
            base_model_id=runtime.base_model_id,
        )
        for segment in range(5):
            step_id = segment * 100
            store.append_event("intervention_start")
            store.append_observation(
                kind="expert", step_id=step_id, images=images,
                state=state, prompt=prompt,
            )
            store.append_observation(
                kind="expert", step_id=step_id + 1, images=images,
                state=expert_actions[0], prompt=prompt,
            )
            store.append_event("intervention_stop")
        episode_dirs.append(store.finish(
            "success",
            control_metrics={
                "policy_steps": 0, "expert_steps": 10,
                "intervention_count": 5,
            },
        ))

    metrics = runtime.train_episodes(
        episode_dirs,
        current_version=0,
        metrics_dir=episode_dirs[0],
    )
    feature = extract_vlm_feature(runtime.policy, observation)
    actor_observation = {
        "pixels": feature[None, :, None],
        "state": state[None, :, None],
    }
    before_reload = runtime.steering.sample_actions(actor_observation)
    runtime.steering = None
    runtime.policy_version = 0
    runtime._restore_active_version()
    after_reload = runtime.steering.sample_actions(actor_observation)
    reload_max_abs = float(np.max(np.abs(before_reload - after_reload)))
    if reload_max_abs > 1e-6:
        raise RuntimeError(f"checkpoint reload mismatch: {reload_max_abs}")

    server = ARXFlowDaggerServer(runtime, output_root=run_root, default_prompt=prompt)
    runtime.steering_eligible = True
    client_session_id = "offline-smoke"
    server_session_id = server.handle_message({"cmd": "health"})[
        "server_session_id"
    ]
    encoded_images = _message_images(images)
    server.handle_message({
        "cmd": "episode_start",
        "episode_id": "smoke_shadow",
        "run_stage": "shadow",
        "shadow_mode": True,
        "prompt": prompt,
        "protocol_version": PROTOCOL_VERSION,
        "client_session_id": client_session_id,
        "server_session_id": server_session_id,
        "base_model_id": runtime.base_model_id,
        "requested_policy_version": runtime.policy_version,
    })
    shadow = server.handle_message({
        "cmd": "predict",
        "episode_id": "smoke_shadow",
        "step_id": 0,
        "policy_version": runtime.policy_version,
        "request_generation": 0,
        "client_session_id": client_session_id,
        "server_session_id": server_session_id,
        "state": state.tolist(),
        **encoded_images,
    })
    server.handle_message({
        "cmd": "episode_end",
        "episode_id": "smoke_shadow",
        "label": "failure",
        "client_session_id": client_session_id,
        "server_session_id": server_session_id,
    })

    server.handle_message({
        "cmd": "episode_start",
        "episode_id": "smoke_closed",
        "run_stage": "closed_loop",
        "shadow_mode": False,
        "prompt": prompt,
        "protocol_version": PROTOCOL_VERSION,
        "client_session_id": client_session_id,
        "server_session_id": server_session_id,
        "base_model_id": runtime.base_model_id,
        "requested_policy_version": runtime.policy_version,
    })
    closed = server.handle_message({
        "cmd": "predict",
        "episode_id": "smoke_closed",
        "step_id": 0,
        "policy_version": runtime.policy_version,
        "request_generation": 0,
        "client_session_id": client_session_id,
        "server_session_id": server_session_id,
        "state": state.tolist(),
        **encoded_images,
    })
    server.handle_message({
        "cmd": "episode_end",
        "episode_id": "smoke_closed",
        "label": "failure",
        "client_session_id": client_session_id,
        "server_session_id": server_session_id,
    })

    report = {
        "passed": True,
        "run_root": str(run_root),
        "training": metrics,
        "checkpoint_reload_max_abs": reload_max_abs,
        "shadow_executed_policy": shadow["executed_policy"],
        "closed_loop_executed_policy": closed["executed_policy"],
        "shadow_action_shape": [len(shadow["actions"]), len(shadow["actions"][0])],
        "closed_action_shape": [len(closed["actions"]), len(closed["actions"][0])],
    }
    report_path = run_root / "end_to_end_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
