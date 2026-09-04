"""Gate and train the first ARX steering policy from recorded success episodes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from arx_adapter import extract_vlm_feature
from arx_campaign import preload_campaign_config, add_arx_runtime_args
from arx_trainer import ARXFlowDaggerRuntime


def _eligible_demonstration(path: Path, metadata: dict) -> tuple[bool, str]:
    label = str(metadata.get("label", ""))
    if metadata.get("task_outcome") != "success" and label not in (
        "assisted_success", "success"
    ):
        return False, "not successful"
    if metadata.get("run_stage") != "demonstration":
        return False, "not a demonstration"
    if int(metadata.get("steering_policy_version", 0)) != 0:
        return False, "steering was active"
    if metadata.get("runtime_mode") == "protocol_only":
        return False, "protocol-only episode"
    steps_path = path / "steps.jsonl"
    if not steps_path.exists():
        return False, "missing steps.jsonl"
    expert = []
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == "policy" and row.get("executed_policy") == "protocol_only":
            return False, "protocol-only episode"
        if row.get("kind") == "expert":
            expert.append(row)
    adjacent = [
        (left, right)
        for left, right in zip(expert, expert[1:])
        if int(right["step_id"]) == int(left["step_id"]) + 1
    ]
    if not adjacent:
        return False, "no adjacent expert-state pair"
    for row in expert:
        images = row.get("images", {})
        if len(images) != 3 or any(not (path / rel).is_file() for rel in images.values()):
            return False, "incomplete three-camera observation"
    return True, "eligible"


def main() -> None:
    cfg = preload_campaign_config()
    parser = argparse.ArgumentParser()
    add_arx_runtime_args(parser, cfg)
    parser.add_argument("--minimum-success-episodes", type=int, default=1)
    parser.add_argument(
        "--minimum-validation-reduction",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    started = time.monotonic()
    def progress(message: str) -> None:
        print(
            f"[FlowDAgger offline][{time.monotonic() - started:8.1f}s] {message}",
            flush=True,
        )

    output_root = Path(args.output_root)
    episode_dirs = []
    rejected = {}
    for path in sorted((output_root / "episodes").glob("*")):
        metadata = json.loads(
            (path / "metadata.json").read_text(encoding="utf-8")
        )
        eligible, reason = _eligible_demonstration(path, metadata)
        if eligible:
            episode_dirs.append(path)
        elif metadata.get("run_stage") == "demonstration":
            rejected[path.name] = reason
    if len(episode_dirs) < args.minimum_success_episodes:
        raise RuntimeError(
            f"need at least {args.minimum_success_episodes} success episodes, "
            f"found {len(episode_dirs)}; rejected={rejected}"
        )

    progress(
        f"数据门禁通过: eligible={len(episode_dirs)} "
        f"minimum={args.minimum_success_episodes} rejected={len(rejected)}"
    )
    progress(f"阶段 0/4 加载基座模型: {args.checkpoint}")

    runtime = ARXFlowDaggerRuntime(
        openpi_root=args.openpi_root,
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        output_root=str(output_root),
        default_prompt=args.default_prompt,
        inversion_batch_size=cfg.offline_inversion_batch_size,
        steering_lr=cfg.steering_lr,
    )
    progress(
        f"基座模型加载完成: base_model_id={runtime.base_model_id} "
        f"active_version={runtime.policy_version}"
    )
    gate_dir = output_root / "offline_training" / time.strftime("%Y%m%d_%H%M%S")
    progress(f"本次训练输出目录: {gate_dir}")
    metrics = runtime.train_episodes(
        episode_dirs,
        runtime.policy_version,
        metrics_dir=gate_dir,
        min_validation_reduction=args.minimum_validation_reduction,
    )

    progress("训练完成，开始 checkpoint 重载一致性检查")
    first_observation, _, _ = next(runtime._load_expert_windows(episode_dirs[0]))
    feature = extract_vlm_feature(runtime.policy, first_observation)
    actor_observation = {
        "pixels": feature[None, :, None],
        "state": np.asarray(first_observation["state"], dtype=np.float32)[None, :, None],
    }
    before_reload = runtime._predict_coefficients(actor_observation)
    runtime.steering = None
    runtime.policy_version = 0
    runtime._restore_active_version()
    after_reload = runtime._predict_coefficients(actor_observation)
    reload_max_abs = float(np.max(np.abs(before_reload - after_reload)))
    if reload_max_abs > 1e-6:
        raise RuntimeError(f"checkpoint reload mismatch: max_abs={reload_max_abs}")
    metrics["checkpoint_reload_max_abs"] = reload_max_abs
    eligibility_path = runtime.mark_closed_loop_eligible(
        runtime.policy_version, metrics=metrics, reload_max_abs=reload_max_abs
    )
    metrics["closed_loop_eligible"] = True
    metrics["eligibility_path"] = str(eligibility_path)
    runtime._atomic_json(gate_dir / "offline_gate.json", metrics)
    progress(
        f"训练完成: version={runtime.policy_version} "
        f"reload_max_abs={reload_max_abs:.3g} total={time.monotonic() - started:.1f}s"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
