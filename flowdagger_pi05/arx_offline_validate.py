"""Reproducible offline gate for the ARX pi0.5 FlowDAgger adapter."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arx_adapter import OpenPIActionTransformAdapter, extract_vlm_feature
from arx_campaign import add_arx_runtime_args, get_campaign_config, preload_campaign_config


def _valid_state() -> np.ndarray:
    state = np.zeros(20, dtype=np.float32)
    state[[0, 10]] = 0.3
    state[[2, 12]] = 0.2
    state[3:9] = state[13:19] = [1, 0, 0, 0, 1, 0]
    state[[9, 19]] = 0.07
    return state


def _observation(state: np.ndarray) -> dict:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "observation/image": image,
        "observation/left_wrist_image": image,
        "observation/right_wrist_image": image,
        "state": np.asarray(state, dtype=np.float32),
        "prompt": get_campaign_config().default_prompt,
    }


def _rotation_matrix(values: np.ndarray) -> np.ndarray:
    first = values[..., :3]
    second = values[..., 3:6]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-9)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-9)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    relative = np.swapaxes(_rotation_matrix(actual.copy()), -1, -2) @ _rotation_matrix(
        expected.copy()
    )
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return float(np.max(np.degrees(np.arccos(cosine))))


def _load_log_states(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = [f"state_{index}" for index in range(20)]
    if not rows or any(column not in rows[0] for column in columns):
        raise ValueError(f"{path} does not contain state_0..state_19")
    values = np.asarray(
        [[float(row[column]) for column in columns] for row in rows[:50]],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains NaN/Inf")
    return values


def _gpu_memory() -> dict[str, float]:
    used = 0.0
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        pid, memory = [item.strip() for item in line.split(",")]
        if int(pid) == os.getpid():
            used += float(memory)
    total = float(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()[0]
    )
    return {"process_used_mib": used, "total_mib": total, "free_margin_mib": total - used}


def main() -> None:
    cfg = preload_campaign_config()
    parser = argparse.ArgumentParser()
    add_arx_runtime_args(parser, cfg)
    parser.add_argument("--action-log", type=Path)
    parser.add_argument("--chunks", type=int, default=20)
    args = parser.parse_args()
    if args.chunks < 20:
        raise ValueError("at least 20 chunks are required for a 95% gate")

    import sys

    root = Path(args.openpi_root)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "packages" / "openpi-client" / "src"))
    from openpi.models import model as model_module
    from openpi.policies import policy_config
    from openpi.training import config
    from flow_matching_inverter import FlowMatchingInverter

    policy = policy_config.create_trained_policy(
        config.get_config(args.config), args.checkpoint, default_prompt=args.default_prompt
    )
    adapter = OpenPIActionTransformAdapter(policy)
    state = _valid_state()
    observation = _observation(state)
    feature = extract_vlm_feature(policy, observation)

    log_states = _load_log_states(args.action_log) if args.action_log else np.repeat(
        state[None, :], 50, axis=0
    )
    log_observation = _observation(log_states[0])
    transformed_log, internal_log = adapter.expert_to_internal(
        log_observation, log_states
    )
    log_roundtrip = adapter.internal_to_env(transformed_log, internal_log)
    log_roundtrip_error = float(
        np.max(np.abs(log_roundtrip[: len(log_states)] - log_states))
    )

    inverter = FlowMatchingInverter(
        policy._model,
        method="perstep_fp",
        num_denoise_steps=10,
        fp_per_step=5,
        seed=42,
    )
    rng = np.random.default_rng(123)
    chunks = []
    started = time.monotonic()
    for index in range(args.chunks):
        known_noise = rng.normal(size=(50, 32)).astype(np.float32)
        expected = np.asarray(policy.infer(observation, noise=known_noise)["actions"])
        transformed, target = adapter.expert_to_internal(observation, expected)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        model_observation = model_module.Observation.from_dict(batched)
        target_batch = jnp.asarray(target)[None, ...]
        recovered, _ = inverter.invert(model_observation, target_batch)
        reconstructed = np.asarray(
            inverter._denoise(model_observation, recovered)[0], dtype=np.float32
        )
        decoded = adapter.internal_to_env(transformed, reconstructed)
        mse = float(np.mean(np.square(reconstructed[:, :20] - target[:, :20])))
        xyz = float(
            np.max(np.abs(decoded[:, [0, 1, 2, 10, 11, 12]] - expected[:, [0, 1, 2, 10, 11, 12]]))
        )
        gripper = float(np.max(np.abs(decoded[:, [9, 19]] - expected[:, [9, 19]])))
        rotation = max(
            _rotation_error_deg(decoded[:, 3:9], expected[:, 3:9]),
            _rotation_error_deg(decoded[:, 13:19], expected[:, 13:19]),
        )
        passed = bool(
            np.all(np.isfinite(reconstructed))
            and mse <= 1e-3
            and xyz <= 0.005
            and rotation <= 5.0
            and gripper <= 0.005
        )
        chunks.append({
            "index": index,
            "normalized_action_mse": mse,
            "xyz_max_m": xyz,
            "rotation_max_deg": rotation,
            "gripper_max_m": gripper,
            "passed": passed,
        })

    memory = _gpu_memory()
    pass_fraction = sum(item["passed"] for item in chunks) / len(chunks)
    report = {
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "vlm_feature_shape": list(feature.shape),
        "log_source": str(args.action_log) if args.action_log else "synthetic",
        "transform_roundtrip_max_abs": log_roundtrip_error,
        "pass_fraction": pass_fraction,
        "elapsed_s": time.monotonic() - started,
        "gpu_memory": memory,
        "chunks": chunks,
    }
    report_dir = Path(args.output_root) / "offline_validation" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if log_roundtrip_error > 1e-5:
        raise RuntimeError(f"20D transform roundtrip failed: {log_roundtrip_error}")
    if pass_fraction < 0.95:
        raise RuntimeError(f"inversion pass fraction failed: {pass_fraction}")
    if memory["process_used_mib"] >= 22 * 1024 or memory["free_margin_mib"] < 2 * 1024:
        raise RuntimeError(f"GPU memory gate failed: {memory}")
    print(report_path)


if __name__ == "__main__":
    main()
