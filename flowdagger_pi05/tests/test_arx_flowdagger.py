import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from arx_adapter import (
    pad_actions_with_policy,
    policy_rows_without_later_intervention,
)
from arx_flowdagger_server import ARXFlowDaggerServer
from arx_episode_store import EpisodeStore
from arx_offline_train import _eligible_demonstration
from arx_trainer import (
    ARXFlowDaggerRuntime,
    ONLINE_BC_STEPS,
    bc_milestone_steps,
    online_batch_mix_sizes,
    online_episode_isolated_split,
    schedule_bc_hyperparams,
    should_freeze_target_normalization,
)
from train_utils import _expand_noise_basis, _project_noise_to_basis


class FakeRuntime:
    def __init__(self):
        self.policy_version = 0
        self.trained = []

    def infer(self, observation, *, shadow_mode):
        return {"actions": np.zeros((50, 20), dtype=np.float32), "executed_policy": "base"}

    def train_episode(self, episode_dir, current_version):
        self.trained.append(episode_dir)
        self.policy_version = current_version + 1
        return {"policy_version": self.policy_version, "num_kept": 1}


def images():
    image = np.zeros((3, 8, 8), dtype=np.uint8).tolist()
    return {
        "observation/image": image,
        "observation/left_wrist_image": image,
        "observation/right_wrist_image": image,
    }


def _frame():
    return {
        key: np.zeros((8, 8, 3), dtype=np.uint8)
        for key in images()
    }


def _moving_state(step: int) -> list[float]:
    state = [0.0] * 20
    offset = step * 0.01
    state[0] = offset
    state[10] = offset
    return state


def test_training_window_uses_pre_intervention_policy_anchor(tmp_path):
    store = EpisodeStore(tmp_path)
    store.start("boundary", run_stage="bootstrap")
    frame = {
        key: np.zeros((8, 8, 3), dtype=np.uint8)
        for key in images()
    }
    policy_sequence = store.append_observation(
        kind="policy", step_id=10, images=frame, state=[0.0] * 20,
        prompt="connect", request_generation=7,
    )
    store.append_event("intervention_start", step_id=11)
    store.append_observation(
        kind="boundary", step_id=11, images=frame, state=[0.1] * 20,
        prompt="connect", request_generation=7,
    )
    for step in range(11, 61):
        store.append_observation(
            kind="expert", step_id=step, images=frame,
            state=_moving_state(step), prompt="connect",
        )
    episode = store.finish("success")
    runtime = object.__new__(ARXFlowDaggerRuntime)
    runtime.default_prompt = "connect"
    window = next(runtime._load_expert_windows(episode))
    assert window[2]["anchor_kind"] == "policy"
    assert window[2]["anchor_record_sequence"] == policy_sequence
    assert window[2]["valid_length"] == 50
    assert window[1].shape == (50, 20)


def test_training_window_strips_switch_holds_and_pads_with_policy(tmp_path):
    store = EpisodeStore(tmp_path)
    store.start("pad-policy", run_stage="bootstrap")
    frame = _frame()
    store.append_observation(
        kind="policy", step_id=10, images=frame, state=[0.0] * 20,
        prompt="connect", request_generation=7,
    )
    store.append_event("intervention_start", step_id=11)
    store.append_observation(
        kind="boundary", step_id=11, images=frame, state=[0.0] * 20,
        prompt="connect", request_generation=7,
    )
    hold = [0.0] * 20
    for step in (11, 12, 13):
        store.append_observation(
            kind="expert", step_id=step, images=frame, state=hold, prompt="connect",
        )
    for step in (14, 15, 16):
        store.append_observation(
            kind="expert", step_id=step, images=frame,
            state=_moving_state(step), prompt="connect",
        )
    for step in (17, 18):
        store.append_observation(
            kind="expert", step_id=step, images=frame,
            state=_moving_state(16), prompt="connect",
        )
    predicted = np.arange(50 * 20, dtype=np.float32).reshape(50, 20) * 0.001
    store.append_event("intervention_stop")
    store.append_observation(
        kind="policy", step_id=19, images=frame, state=_moving_state(16),
        prompt="connect", predicted_actions=predicted.tolist(),
    )
    episode = store.finish("success")
    runtime = object.__new__(ARXFlowDaggerRuntime)
    runtime.default_prompt = "connect"
    assert list(runtime._load_expert_windows(episode)) == []
    future = [
        {"state": _moving_state(step), "timestamp_s": step / 30.0}
        for step in (14, 15, 16)
    ]
    policy_rows = [{
        "state": _moving_state(16),
        "timestamp_s": 19 / 30.0,
        "predicted_actions": predicted.tolist(),
    }]
    actions, info = pad_actions_with_policy(future, policy_rows)
    assert actions.shape == (50, 20)
    assert info["valid_length"] == 3
    assert info["policy_padded_length"] == 47
    assert info["padded_length"] == 0
    assert info["pad_source"] == "predicted_actions"
    np.testing.assert_allclose(actions[:3], [_moving_state(step) for step in (14, 15, 16)])
    np.testing.assert_allclose(actions[3:], predicted[:47])


def test_short_expert_window_resamples_following_policy_states(tmp_path):
    store = EpisodeStore(tmp_path)
    store.start("pad-states", run_stage="bootstrap")
    frame = _frame()
    store.append_observation(
        kind="policy", step_id=0, images=frame, state=[0.0] * 20,
        prompt="connect", request_generation=1,
    )
    store.append_event("intervention_start", step_id=1)
    store.append_observation(
        kind="boundary", step_id=1, images=frame, state=[0.0] * 20,
        prompt="connect", request_generation=1,
    )
    for step in (1, 2):
        store.append_observation(
            kind="expert", step_id=step, images=frame,
            state=_moving_state(step), prompt="connect",
        )
    store.append_event("intervention_stop")
    later = _moving_state(40)
    store.append_observation(
        kind="policy", step_id=20, images=frame, state=later, prompt="connect",
    )
    episode = store.finish("success")
    runtime = object.__new__(ARXFlowDaggerRuntime)
    runtime.default_prompt = "connect"
    assert list(runtime._load_expert_windows(episode)) == []
    future = [
        {"state": _moving_state(step), "timestamp_s": step / 30.0}
        for step in (1, 2)
    ]
    policy_rows = [{"state": later, "timestamp_s": 20 / 30.0}]
    actions, info = pad_actions_with_policy(future, policy_rows)
    assert info["pad_source"] == "policy_states"
    assert info["policy_padded_length"] > 0
    assert actions.shape[0] == info["valid_length"] + info["policy_padded_length"]
    np.testing.assert_allclose(actions[-1], later, atol=1e-5)


def test_server_skips_leading_and_trailing_holds_but_keeps_interior(tmp_path):
    runtime = FakeRuntime()
    server = ARXFlowDaggerServer(
        runtime, output_root=tmp_path, default_prompt="connect",
    )
    server.handle_message({
        "cmd": "episode_start", "episode_id": "holds", "shadow_mode": True,
    })
    policy = [0.0] * 20
    server.handle_message({
        "cmd": "predict", "episode_id": "holds", "step_id": 0,
        "state": policy, **images(),
    })
    server.handle_message({"cmd": "intervention_start", "episode_id": "holds"})
    for step in (1, 2, 3):
        result = server.handle_message({
            "cmd": "expert_step", "episode_id": "holds", "step_id": step,
            "state": policy, **images(),
        })
        assert result.get("skipped_pause") is True
    for step in (4, 5):
        result = server.handle_message({
            "cmd": "expert_step", "episode_id": "holds", "step_id": step,
            "state": _moving_state(step), **images(),
        })
        assert "skipped_pause" not in result
        assert "buffered_hold" not in result
    interior_hold = _moving_state(5)
    for step in (6, 7):
        result = server.handle_message({
            "cmd": "expert_step", "episode_id": "holds", "step_id": step,
            "state": interior_hold, **images(),
        })
        assert result.get("buffered_hold") is True
    result = server.handle_message({
        "cmd": "expert_step", "episode_id": "holds", "step_id": 8,
        "state": _moving_state(8), **images(),
    })
    assert "buffered_hold" not in result
    trailing = _moving_state(8)
    for step in (9, 10):
        result = server.handle_message({
            "cmd": "expert_step", "episode_id": "holds", "step_id": step,
            "state": trailing, **images(),
        })
        assert result.get("buffered_hold") is True
    server.handle_message({"cmd": "intervention_stop", "episode_id": "holds"})
    result = server.handle_message({
        "cmd": "episode_end", "episode_id": "holds", "label": "abort",
    })
    steps = [
        json.loads(line)
        for line in (Path(result["episode_dir"]) / "steps.jsonl").read_text().splitlines()
    ]
    experts = [row for row in steps if row["kind"] == "expert"]
    assert [row["step_id"] for row in experts] == [4, 5, 6, 7, 8]
    policy_row = next(row for row in steps if row["kind"] == "policy")
    assert np.asarray(policy_row["predicted_actions"]).shape == (50, 20)


def test_steering_coefficients_are_clipped_then_denormalized():
    class FakeSteering:
        def sample_actions(self, observation):
            return np.array([[[-3.0], [0.5], [4.0]]], dtype=np.float32)

    runtime = object.__new__(ARXFlowDaggerRuntime)
    runtime.steering = FakeSteering()
    runtime.noise_basis_k = 3
    # Keep the synthetic internal dimension at one for this focused unit test.
    runtime.target_mean = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    runtime.target_std = np.array([[2.0], [4.0], [5.0]], dtype=np.float32)
    runtime.target_clip_low = np.array([[-2.0], [-1.0], [-3.0]], dtype=np.float32)
    runtime.target_clip_high = np.array([[2.0], [1.0], [3.0]], dtype=np.float32)
    # _predict_coefficients uses the fixed ARX internal width. Expand the
    # synthetic values across that width and check representative entries.
    runtime.noise_basis_k = 3
    runtime.steering.sample_actions = lambda observation: np.tile(
        np.array([[-3.0], [0.5], [4.0]], dtype=np.float32), (1, 32)
    )[None]
    runtime.target_mean = np.tile(runtime.target_mean, (1, 32))
    runtime.target_std = np.tile(runtime.target_std, (1, 32))
    runtime.target_clip_low = np.tile(runtime.target_clip_low, (1, 32))
    runtime.target_clip_high = np.tile(runtime.target_clip_high, (1, 32))
    coefficients = runtime._predict_coefficients({})
    np.testing.assert_allclose(coefficients[:, 0], [6.0, 22.0, 45.0])


def test_assisted_success_episode_is_atomic_and_queues_training(tmp_path):
    runtime = FakeRuntime()
    runtime.policy_version = 1
    server = ARXFlowDaggerServer(
        runtime,
        output_root=tmp_path,
        default_prompt="connect",
    )
    assert server.handle_message({
        "cmd": "episode_start",
        "episode_id": "ep-1",
        "run_stage": "closed_loop",
        "shadow_mode": False,
    })["status"] == "ok"
    server.handle_message({"cmd": "intervention_start", "episode_id": "ep-1"})
    for step in (1, 2):
        server.handle_message({
            "cmd": "expert_step",
            "episode_id": "ep-1",
            "step_id": step,
            "state": _moving_state(step),
            **images(),
        })
    result = server.handle_message({
        "cmd": "episode_end", "episode_id": "ep-1", "label": "assisted_success",
    })
    assert result["training_queued"]
    episode_dirs = list((tmp_path / "episodes").iterdir())
    assert len(episode_dirs) == 1
    final = episode_dirs[0]
    assert re.fullmatch(r"episode_\d{8}_0001_\d{6}", final.name)
    assert final.exists()
    assert not (tmp_path / ".partial" / "ep-1").exists()
    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["label"] == "assisted_success"
    assert metadata["episode_id"] == final.name
    assert metadata["source_episode_id"] == "ep-1"
    for _ in range(100):
        if server.trainer.snapshot()["state"] != "running":
            break
        time.sleep(0.01)
    assert server.trainer.snapshot()["state"] == "succeeded"
    assert runtime.policy_version == 2


def test_episode_folder_sequence_counts_legacy_episode_dirs(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True)
    (episodes / "episode_0001_legacy").mkdir()
    (episodes / "network_validation_123").mkdir()
    runtime = FakeRuntime()
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    server.handle_message({
        "cmd": "episode_start", "episode_id": "internal-id", "shadow_mode": True,
    })
    result = server.handle_message({
        "cmd": "episode_end", "episode_id": "internal-id", "label": "failure",
    })
    final = result["episode_dir"].split("/")[-1]
    assert re.fullmatch(r"episode_\d{8}_0002_\d{6}", final)


@pytest.mark.parametrize("run_stage", ["demonstration", "baseline", "shadow"])
def test_non_closed_loop_success_is_archived_without_training(tmp_path, run_stage):
    runtime = FakeRuntime()
    if run_stage == "shadow":
        runtime.policy_version = 1
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    result = server.handle_message({
        "cmd": "episode_start",
        "episode_id": f"ep-{run_stage}",
        "run_stage": run_stage,
        "shadow_mode": True,
    })
    assert result["status"] == "ok"
    result = server.handle_message({
        "cmd": "episode_end",
        "episode_id": f"ep-{run_stage}",
        "label": "assisted_success",
    })
    assert result["training_queued"] is False
    assert runtime.trained == []


def test_failure_episode_is_archived_without_training(tmp_path):
    runtime = FakeRuntime()
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    server.handle_message({"cmd": "episode_start", "episode_id": "ep-2", "shadow_mode": True})
    result = server.handle_message({
        "cmd": "episode_end", "episode_id": "ep-2", "label": "failure",
    })
    assert result["training_queued"] is False
    assert runtime.trained == []


def test_predict_rejects_stale_policy_version(tmp_path):
    runtime = FakeRuntime()
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    server.handle_message({"cmd": "episode_start", "episode_id": "ep-3", "shadow_mode": True})
    with pytest.raises(RuntimeError, match="policy version mismatch"):
        server.handle_message({
            "cmd": "predict",
            "episode_id": "ep-3",
            "step_id": 0,
            "policy_version": 99,
            "state": [0.0] * 20,
            **images(),
        })


def test_failed_training_keeps_previous_version(tmp_path):
    class FailingRuntime(FakeRuntime):
        def train_episode(self, episode_dir, current_version):
            raise RuntimeError("inversion failed")

    runtime = FailingRuntime()
    runtime.policy_version = 1
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    server.handle_message({
        "cmd": "episode_start",
        "episode_id": "ep-4",
        "run_stage": "closed_loop",
        "shadow_mode": False,
    })
    server.handle_message({"cmd": "intervention_start", "episode_id": "ep-4"})
    for step in (1, 2):
        server.handle_message({
            "cmd": "expert_step", "episode_id": "ep-4", "step_id": step,
            "state": _moving_state(step), **images(),
        })
    server.handle_message({
        "cmd": "episode_end",
        "episode_id": "ep-4",
        "label": "assisted_success",
    })
    for _ in range(100):
        if server.trainer.snapshot()["state"] != "running":
            break
        time.sleep(0.01)
    status = server.trainer.snapshot()
    assert status["state"] == "failed"
    assert status["policy_version"] == 1
    assert runtime.policy_version == 1


def test_episode_metadata_aggregates_shadow_and_control_metrics(tmp_path):
    class ShadowRuntime(FakeRuntime):
        def infer(self, observation, *, shadow_mode):
            base = np.zeros((50, 20), dtype=np.float32)
            return {
                "actions": base,
                "shadow_actions": np.ones((50, 20), dtype=np.float32) * 0.1,
                "executed_policy": "base_shadow",
            }

    server = ARXFlowDaggerServer(
        ShadowRuntime(), output_root=tmp_path, default_prompt="connect"
    )
    server.handle_message({
        "cmd": "episode_start", "episode_id": "ep-5", "shadow_mode": True,
    })
    server.handle_message({
        "cmd": "predict",
        "episode_id": "ep-5",
        "step_id": 0,
        "policy_version": 0,
        "state": [0.0] * 20,
        **images(),
    })
    result = server.handle_message({
        "cmd": "episode_end",
        "episode_id": "ep-5",
        "label": "failure",
        "control_metrics": {
            "policy_steps": 8,
            "expert_steps": 2,
            "eef_step_clamps": 4,
        },
    })
    metadata = json.loads((Path(result["episode_dir"]) / "metadata.json").read_text())
    metrics = metadata["episode_metrics"]
    assert metrics["shadow_observations"] == 1
    assert metrics["shadow_action_mse_mean"] == pytest.approx(0.01)
    assert metrics["control"]["intervention_step_fraction"] == pytest.approx(0.2)
    assert metrics["control"]["eef_clamp_trigger_rate"] == pytest.approx(0.5)


def test_control_metrics_reject_nan(tmp_path):
    server = ARXFlowDaggerServer(
        FakeRuntime(), output_root=tmp_path, default_prompt="connect"
    )
    server.handle_message({"cmd": "episode_start", "episode_id": "ep-6", "shadow_mode": True})
    with pytest.raises(ValueError, match="finite"):
        server.handle_message({
            "cmd": "episode_end",
            "episode_id": "ep-6",
            "label": "abort",
            "control_metrics": {"safety_rejections": float("nan")},
        })


def test_closed_loop_stage_requires_trained_steering_and_shadow_off(tmp_path):
    runtime = FakeRuntime()
    server = ARXFlowDaggerServer(
        runtime, output_root=tmp_path, default_prompt="connect"
    )
    with pytest.raises(RuntimeError, match="trained steering"):
        server.handle_message({
            "cmd": "episode_start",
            "episode_id": "ep-7",
            "run_stage": "closed_loop",
            "shadow_mode": False,
        })

    runtime.policy_version = 1
    result = server.handle_message({
        "cmd": "episode_start",
        "episode_id": "ep-8",
        "run_stage": "closed_loop",
        "shadow_mode": False,
    })
    assert result["policy_version"] == 1


def test_raw_expert_jpeg_preserves_native_resolution(tmp_path):
    server = ARXFlowDaggerServer(
        FakeRuntime(), output_root=tmp_path, default_prompt="connect"
    )
    server.handle_message({
        "cmd": "episode_start",
        "episode_id": "ep-raw",
        "shadow_mode": True,
    })
    server.handle_message({"cmd": "intervention_start", "episode_id": "ep-raw"})
    raw_images = {}
    for key in (
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
    ):
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
        assert ok
        raw_images[key] = encoded.tobytes()
    server.handle_message({
        "cmd": "expert_step",
        "episode_id": "ep-raw",
        "step_id": 0,
        "state": [0.0] * 20,
        "raw_images": raw_images,
    })
    result = server.handle_message({
        "cmd": "episode_end", "episode_id": "ep-raw", "label": "abort",
    })
    row = json.loads(
        (Path(result["episode_dir"]) / "steps.jsonl").read_text().strip()
    )
    assert row["image_shapes"]["observation/image"] == [48, 64, 3]


def test_offline_training_rejects_protocol_only_success(tmp_path):
    (tmp_path / "steps.jsonl").write_text(
        json.dumps({"kind": "policy", "executed_policy": "protocol_only"}) + "\n",
        encoding="utf-8",
    )
    eligible, reason = _eligible_demonstration(tmp_path, {
        "label": "assisted_success",
        "run_stage": "demonstration",
        "steering_policy_version": 0,
    })
    assert eligible is False
    assert reason == "protocol-only episode"


def test_offline_training_accepts_complete_30hz_expert_pair(tmp_path):
    images = {}
    for index in range(3):
        path = tmp_path / f"camera_{index}.jpg"
        path.write_bytes(b"jpeg-placeholder")
        images[f"camera_{index}"] = path.name
    rows = [
        {
            "kind": "expert",
            "step_id": step,
            "timestamp_s": step / 30.0,
            "images": images,
        }
        for step in (10, 11)
    ]
    (tmp_path / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    eligible, reason = _eligible_demonstration(tmp_path, {
        "label": "assisted_success",
        "run_stage": "demonstration",
        "steering_policy_version": 0,
        "runtime_mode": "base_record",
    })
    assert eligible is True
    assert reason == "eligible"


def test_health_reports_fixed_arx_contract(tmp_path):
    server = ARXFlowDaggerServer(FakeRuntime(), output_root=tmp_path, default_prompt="connect")
    health = server.handle_message({"cmd": "health"})
    assert health["action_horizon"] == 50
    assert health["action_dim"] == health["state_dim"] == 20
    assert len(health["camera_keys"]) == 3
    assert health["active_episode_id"] is None
    assert health["stage_counts"]["demonstration"] == {"total": 0, "success": 0}


def test_real_runtime_modes_reject_wrong_collection_stage(tmp_path):
    flow_runtime = FakeRuntime()
    flow_runtime.runtime_mode = "flowdagger"
    flow_runtime.policy_version = 1
    flow_server = ARXFlowDaggerServer(
        flow_runtime, output_root=tmp_path / "flow", default_prompt="connect"
    )
    with pytest.raises(RuntimeError, match="record-only"):
        flow_server.handle_message({
            "cmd": "episode_start", "episode_id": "wrong-demo",
            "run_stage": "demonstration", "shadow_mode": True,
        })

    base_runtime = FakeRuntime()
    base_runtime.runtime_mode = "base_record"
    base_server = ARXFlowDaggerServer(
        base_runtime, output_root=tmp_path / "base", default_prompt="connect"
    )
    with pytest.raises(RuntimeError, match="full FlowDAgger"):
        base_server.handle_message({
            "cmd": "episode_start", "episode_id": "wrong-shadow",
            "run_stage": "shadow", "shadow_mode": True,
        })


def test_deployment_runtime_rejects_base_and_protocol_mismatch(tmp_path):
    runtime = FakeRuntime()
    runtime.base_model_id = "connect_elevator_pins_arx_0901:20000:test"
    runtime.steering_eligible = False
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    common = {
        "cmd": "episode_start", "episode_id": "strict", "shadow_mode": True,
        "client_session_id": "client", "requested_policy_version": 0,
    }
    with pytest.raises(RuntimeError, match="protocol_version"):
        server.handle_message({**common, "base_model_id": runtime.base_model_id})
    with pytest.raises(RuntimeError, match="base model mismatch"):
        server.handle_message({
            **common, "protocol_version": 3, "base_model_id": "wrong",
        })


def test_predict_echoes_request_generation_and_sessions(tmp_path):
    runtime = FakeRuntime()
    runtime.base_model_id = "connect_elevator_pins_arx_0901:20000:test"
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    server_session_id = server.handle_message({"cmd": "health"})["server_session_id"]
    started = server.handle_message({
        "cmd": "episode_start", "episode_id": "echo", "shadow_mode": True,
        "protocol_version": 3, "client_session_id": "client-a",
        "base_model_id": runtime.base_model_id, "requested_policy_version": 0,
        "server_session_id": server_session_id,
    })
    result = server.handle_message({
        "cmd": "predict", "episode_id": "echo", "step_id": 7,
        "request_generation": 3, "client_session_id": "client-a",
        "server_session_id": started["server_session_id"],
        "policy_version": 0, "state": [0.0] * 20, **images(),
    })
    assert result["episode_id"] == "echo"
    assert result["step_id"] == 7
    assert result["request_generation"] == 3
    assert result["server_session_id"] == started["server_session_id"]


def test_closed_loop_success_without_expert_data_does_not_train(tmp_path):
    runtime = FakeRuntime()
    runtime.policy_version = 2
    runtime.steering_eligible = True
    server = ARXFlowDaggerServer(runtime, output_root=tmp_path, default_prompt="connect")
    server.handle_message({
        "cmd": "episode_start", "episode_id": "no-expert",
        "run_stage": "closed_loop", "shadow_mode": False,
    })
    result = server.handle_message({
        "cmd": "episode_end",
        "episode_id": "no-expert",
        "task_outcome": "success",
    })
    assert result["training_queued"] is False
    assert runtime.trained == []
    metadata = json.loads(
        (Path(result["episode_dir"]) / "metadata.json").read_text()
    )
    assert metadata["completion_mode"] == "autonomous"


def test_success_with_intervention_is_automatically_assisted(tmp_path):
    runtime = FakeRuntime()
    runtime.policy_version = 2
    runtime.steering_eligible = True
    server = ARXFlowDaggerServer(
        runtime, output_root=tmp_path, default_prompt="connect"
    )
    server.handle_message({
        "cmd": "episode_start",
        "episode_id": "autonomous",
        "run_stage": "closed_loop",
        "shadow_mode": False,
    })
    server.handle_message({
        "cmd": "intervention_start", "episode_id": "autonomous"
    })
    for step in (1, 2):
        server.handle_message({
            "cmd": "expert_step",
            "episode_id": "autonomous",
            "step_id": step,
            "state": _moving_state(step),
            **images(),
        })
    result = server.handle_message({
        "cmd": "episode_end",
        "episode_id": "autonomous",
        "task_outcome": "success",
    })
    assert result["training_queued"] is True
    metadata = json.loads(
        (Path(result["episode_dir"]) / "metadata.json").read_text()
    )
    assert metadata["task_outcome"] == "success"
    assert metadata["completion_mode"] == "assisted"


def test_assisted_success_requires_consecutive_expert_steps(tmp_path):
    runtime = FakeRuntime()
    runtime.policy_version = 2
    runtime.steering_eligible = True
    server = ARXFlowDaggerServer(
        runtime, output_root=tmp_path, default_prompt="connect"
    )
    server.handle_message({
        "cmd": "episode_start",
        "episode_id": "nonconsecutive",
        "run_stage": "closed_loop",
        "shadow_mode": False,
    })
    server.handle_message({
        "cmd": "intervention_start", "episode_id": "nonconsecutive"
    })
    for step in (1, 3):
        server.handle_message({
            "cmd": "expert_step",
            "episode_id": "nonconsecutive",
            "step_id": step,
            "state": [0.0] * 20,
            **images(),
        })
    result = server.handle_message({
        "cmd": "episode_end",
        "episode_id": "nonconsecutive",
        "label": "assisted_success",
    })
    assert result["training_queued"] is False
    assert runtime.trained == []


def test_server_restart_changes_session_id(tmp_path):
    first = ARXFlowDaggerServer(FakeRuntime(), output_root=tmp_path / "a", default_prompt="x")
    second = ARXFlowDaggerServer(FakeRuntime(), output_root=tmp_path / "b", default_prompt="x")
    assert (
        first.handle_message({"cmd": "health"})["server_session_id"]
        != second.handle_message({"cmd": "health"})["server_session_id"]
    )


def test_bc_schedule_follows_50_epochs_and_min_batch_64():
    assert schedule_bc_hyperparams(32) == (32, 50)
    assert schedule_bc_hyperparams(64) == (64, 50)
    assert schedule_bc_hyperparams(65) == (64, 100)
    assert schedule_bc_hyperparams(193) == (64, 200)


def test_online_batch_mix_is_4_4_2():
    assert ONLINE_BC_STEPS == 100
    assert online_batch_mix_sizes(64, 10, 10, 10) == (26, 26, 12)
    assert online_batch_mix_sizes(64, 10, 0, 10) == (43, 0, 21)
    assert online_batch_mix_sizes(64, 10, 0, 0) == (64, 0, 0)
    assert online_batch_mix_sizes(1, 10, 10, 10) == (1, 0, 0)


def test_normalization_freezes_after_three_success_episodes():
    assert should_freeze_target_normalization(1, False) is False
    assert should_freeze_target_normalization(2, False) is False
    assert should_freeze_target_normalization(3, False) is True
    assert should_freeze_target_normalization(1, True) is True


def test_full_horizon_noise_skips_dct():
    noise = np.linspace(-1.0, 1.0, 50 * 32, dtype=np.float32).reshape(50, 32)
    projected = _project_noise_to_basis(noise, 50)
    np.testing.assert_allclose(projected, noise)
    expanded = _expand_noise_basis(projected, 50)
    np.testing.assert_allclose(expanded[0], noise)


def test_bc_milestones_are_evenly_spaced_and_at_least_five():
    steps = bc_milestone_steps(150)
    assert steps[0] == 0
    assert steps[-1] == 150
    assert len(steps) >= 5
    assert steps == [0, 38, 75, 112, 150]
    assert bc_milestone_steps(3) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        schedule_bc_hyperparams(0)


def test_autonomous_windows_use_non_intervention_policy_chunks(tmp_path):
    store = EpisodeStore(tmp_path)
    store.start("auto-chunks", run_stage="closed_loop")
    frame = _frame()
    predicted = np.arange(50 * 20, dtype=np.float32).reshape(50, 20) * 0.001
    store.append_observation(
        kind="policy", step_id=0, images=frame, state=[0.0] * 20,
        prompt="connect", request_generation=1, predicted_actions=predicted.tolist(),
    )
    store.append_event("intervention_start", step_id=1)
    store.append_observation(
        kind="boundary", step_id=1, images=frame, state=[0.1] * 20,
        prompt="connect", request_generation=1,
    )
    store.append_observation(
        kind="expert", step_id=1, images=frame, state=_moving_state(1), prompt="connect",
    )
    store.append_event("intervention_stop")
    later = predicted + 0.5
    store.append_observation(
        kind="policy", step_id=20, images=frame, state=_moving_state(20),
        prompt="connect", predicted_actions=later.tolist(),
    )
    episode = store.finish("success")
    runtime = object.__new__(ARXFlowDaggerRuntime)
    runtime.default_prompt = "connect"
    windows = list(runtime._load_autonomous_windows(episode))
    assert len(windows) == 1
    np.testing.assert_allclose(windows[0][1], later)


def test_policy_rows_without_later_intervention_keeps_trailing_and_autonomous():
    rows = [
        {"kind": "policy", "id": "before"},
        {"kind": "boundary"},
        {"kind": "expert"},
        {"kind": "policy", "id": "after"},
    ]
    kept = policy_rows_without_later_intervention(rows)
    assert [row["id"] for row in kept] == ["after"]
    autonomous = [
        {"kind": "policy", "id": "a"},
        {"kind": "policy", "id": "b"},
    ]
    assert policy_rows_without_later_intervention(autonomous) == autonomous
    assert policy_rows_without_later_intervention(
        [{"kind": "policy", "id": "only"}, {"kind": "expert"}]
    ) == []


def test_online_split_holds_out_entire_current_episode():
    origins = np.array(
        ["history", "history", "current", "prior", "current", "prior"]
    )
    train, val = online_episode_isolated_split(origins)
    assert list(train) == [0, 1, 3, 5]
    assert list(val) == [2, 4]
    assert not set(train.tolist()) & set(val.tolist())


def test_autonomous_success_keeps_all_policy_predicted_actions(tmp_path):
    store = EpisodeStore(tmp_path)
    store.start("auto-success", run_stage="closed_loop")
    frame = _frame()
    predicted = np.arange(50 * 20, dtype=np.float32).reshape(50, 20) * 0.001
    store.append_observation(
        kind="policy", step_id=0, images=frame, state=[0.0] * 20,
        prompt="connect", predicted_actions=predicted.tolist(),
    )
    later = predicted + 0.25
    store.append_observation(
        kind="policy", step_id=1, images=frame, state=_moving_state(1),
        prompt="connect", predicted_actions=later.tolist(),
    )
    episode = store.finish("success")
    runtime = object.__new__(ARXFlowDaggerRuntime)
    runtime.default_prompt = "connect"
    windows = list(runtime._load_autonomous_windows(episode))
    assert len(windows) == 2
    np.testing.assert_allclose(windows[0][1], predicted)
    np.testing.assert_allclose(windows[1][1], later)
