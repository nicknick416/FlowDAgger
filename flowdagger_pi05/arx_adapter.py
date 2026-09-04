"""ARX bimanual observation and action adapters for FlowDAgger."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from arx_campaign import get_campaign_config


CAMERA_KEYS = (
    "observation/image",
    "observation/left_wrist_image",
    "observation/right_wrist_image",
)
STATE_DIM = 20
ACTION_DIM = 20
INTERNAL_ACTION_DIM = 32
ACTION_HORIZON = 50
# Intervention windows shorter than the action horizon are mostly policy pad.
MIN_INTERVENTION_VALID_LENGTH = ACTION_HORIZON
PROTOCOL_VERSION = 3
# Switch-delay holds are typically <0.5 mm/arm at 30 Hz. Interior teleop
# micro-motions are kept; only prefix/suffix runs at a mode boundary are trimmed.
PAUSE_POSITION_M = 5e-4
PAUSE_GRIPPER = 1e-3
_cfg = get_campaign_config()
BASE_CHECKPOINT_NAME = _cfg.base_checkpoint_name
BASE_CHECKPOINT_STEP = _cfg.base_checkpoint_step
BASE_CHECKPOINT = _cfg.base_checkpoint
BASE_ASSET_ID = _cfg.base_asset_id
CAMPAIGN_ID = _cfg.campaign_id
DEFAULT_OUTPUT_ROOT = _cfg.output_root
RAW_DATA_ROOT = _cfg.raw_data_root
DEMO_BUFFER_DIR = _cfg.demo_buffer_dir
DEMO_BUFFER_COUNT = _cfg.demo_buffer_count
WINDOW_STRIDE = _cfg.window_stride


def resolve_base_model_identity(checkpoint_dir: str | Path) -> dict[str, str]:
    """Validate the immutable deployment base and return a stable identity."""
    cfg = get_campaign_config()
    checkpoint = Path(checkpoint_dir).resolve()
    expected = Path(cfg.base_checkpoint).resolve()
    if checkpoint != expected:
        raise ValueError(f"checkpoint must be {expected}, got {checkpoint}")
    norm_stats = checkpoint / "assets" / cfg.base_asset_id / "norm_stats.json"
    metadata = checkpoint / "_CHECKPOINT_METADATA"
    if not norm_stats.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            f"checkpoint is missing metadata or {cfg.base_asset_id} norm stats"
        )
    digest = hashlib.sha256()
    for path in (metadata, norm_stats):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "base_model_id": (
            f"{cfg.base_checkpoint_name}:{cfg.base_checkpoint_step}:"
            f"{digest.hexdigest()[:16]}"
        ),
        "checkpoint": str(checkpoint),
        "asset_id": cfg.base_asset_id,
    }


def validate_arx_state(state: Any) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32).reshape(-1)
    if value.shape != (STATE_DIM,):
        raise ValueError(f"ARX FlowDAgger state must be 20D, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("ARX FlowDAgger state contains NaN/Inf")
    return value


def is_control_hold(
    previous_state: Any,
    current_state: Any,
    *,
    position_m: float = PAUSE_POSITION_M,
    gripper: float = PAUSE_GRIPPER,
) -> bool:
    """True when both arms and grippers barely moved (switch-delay pause)."""
    previous = np.asarray(previous_state, dtype=np.float64).reshape(-1)
    current = np.asarray(current_state, dtype=np.float64).reshape(-1)
    if previous.shape != (STATE_DIM,) or current.shape != (STATE_DIM,):
        return False
    left = float(np.linalg.norm(previous[:3] - current[:3]))
    right = float(np.linalg.norm(previous[10:13] - current[10:13]))
    grip = abs(float(previous[9] - current[9])) + abs(float(previous[19] - current[19]))
    return left < position_m and right < position_m and grip < gripper


def trim_boundary_holds(
    rows: list[dict[str, Any]],
    *,
    previous_state: Any | None = None,
) -> list[dict[str, Any]]:
    """Drop leading/trailing hold frames at a Human↔Policy boundary.

    Interior holds (human pausing mid-correction) are kept. Only the
    contiguous prefix after a mode switch and the contiguous suffix before
    the next switch are removed.
    """
    if not rows:
        return []
    start = 0
    while start < len(rows):
        previous = (
            previous_state if start == 0 and previous_state is not None
            else rows[start - 1]["state"] if start > 0 else None
        )
        if previous is None or not is_control_hold(previous, rows[start]["state"]):
            break
        start += 1
    end = len(rows)
    while end - start > 1 and is_control_hold(
        rows[end - 2]["state"], rows[end - 1]["state"]
    ):
        end -= 1
    if end - start == 1 and previous_state is not None and is_control_hold(
        previous_state, rows[start]["state"]
    ):
        return []
    return rows[start:end]


def policy_rows_without_later_intervention(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep policy rows from segments that are not followed by an expert takeover.

    Assisted successes typically look like policy → expert → policy. The first
    policy segment caused the intervention, so its predicted_actions are not
    autonomous positives. The trailing policy segment, and every policy row in
    an autonomous-success episode, are kept.
    """
    later_expert = False
    keep = [False] * len(rows)
    for index in range(len(rows) - 1, -1, -1):
        kind = rows[index].get("kind")
        if kind == "policy":
            keep[index] = not later_expert
        elif kind == "expert":
            later_expert = True
    return [row for index, row in enumerate(rows) if keep[index]]


def median_step_dt(rows: list[dict[str, Any]], default: float = 1.0 / 30.0) -> float:
    if len(rows) < 2:
        return default
    deltas = np.diff([float(row["timestamp_s"]) for row in rows])
    deltas = deltas[np.isfinite(deltas) & (deltas > 1e-4)]
    if not len(deltas):
        return default
    return float(np.median(deltas))


def resample_states(
    rows: list[dict[str, Any]],
    *,
    t_start: float,
    dt: float,
    count: int,
) -> np.ndarray:
    """Linearly interpolate recorded states onto a regular action grid."""
    if count <= 0 or not rows:
        return np.zeros((0, STATE_DIM), dtype=np.float32)
    times = np.asarray([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
    states = np.asarray([row["state"] for row in rows], dtype=np.float32)
    order = np.argsort(times)
    times = times[order]
    states = states[order]
    targets = t_start + dt * np.arange(count, dtype=np.float64)
    targets = targets[targets <= times[-1] + dt * 0.25]
    if not len(targets):
        return np.zeros((0, STATE_DIM), dtype=np.float32)
    out = np.empty((len(targets), states.shape[1]), dtype=np.float32)
    for index, time_s in enumerate(targets):
        if time_s <= times[0]:
            out[index] = states[0]
            continue
        if time_s >= times[-1]:
            out[index] = states[-1]
            continue
        right = int(np.searchsorted(times, time_s, side="right"))
        left = max(right - 1, 0)
        right = min(max(right, 1), len(times) - 1)
        left = min(left, right - 1)
        span = times[right] - times[left]
        weight = 0.0 if span <= 1e-9 else (time_s - times[left]) / span
        out[index] = states[left] * (1.0 - weight) + states[right] * weight
    return out


def predicted_actions_from_row(row: Mapping[str, Any]) -> np.ndarray | None:
    value = row.get("predicted_actions")
    if value is None:
        return None
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not len(actions):
        return None
    return actions


def pad_actions_with_policy(
    future_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    *,
    action_horizon: int = ACTION_HORIZON,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pad a short expert horizon with the following policy trajectory.

    Prefers the next policy's stored `predicted_actions` chunk (native 50-step
    rate). Falls back to time-resampled executed policy states. Remaining
    shortfall is left for last-frame hold padding in the action adapter.
    """
    expert_actions = [
        np.asarray(row["state"], dtype=np.float32) for row in future_rows
    ]
    if len(expert_actions) >= action_horizon:
        actions = np.stack(expert_actions[:action_horizon], axis=0)
        return actions, {
            "valid_length": int(action_horizon),
            "policy_padded_length": 0,
            "padded_length": 0,
            "pad_source": "none",
        }
    need = action_horizon - len(expert_actions)
    pad_chunks: list[np.ndarray] = []
    pad_source = "none"
    for row in policy_rows:
        predicted = predicted_actions_from_row(row)
        if predicted is None:
            continue
        pad_chunks.append(predicted[:need])
        pad_source = "predicted_actions"
        break
    filled = 0 if not pad_chunks else int(pad_chunks[0].shape[0])
    if filled < need and policy_rows:
        dt = median_step_dt(future_rows)
        t_start = (
            float(future_rows[-1]["timestamp_s"]) + dt
            if future_rows else float(policy_rows[0]["timestamp_s"])
        )
        moving_policy = trim_boundary_holds(
            policy_rows,
            previous_state=future_rows[-1]["state"] if future_rows else None,
        )
        resampled = resample_states(
            moving_policy, t_start=t_start, dt=dt, count=need - filled
        )
        if len(resampled):
            pad_chunks.append(resampled)
            pad_source = (
                "predicted_actions+policy_states"
                if pad_source == "predicted_actions"
                else "policy_states"
            )
    if pad_chunks:
        pad = np.concatenate(pad_chunks, axis=0)[:need]
    else:
        pad = np.zeros((0, ACTION_DIM), dtype=np.float32)
    if expert_actions and len(pad):
        actions = np.concatenate(
            [np.stack(expert_actions, axis=0), pad], axis=0
        )
    elif expert_actions:
        actions = np.stack(expert_actions, axis=0)
    else:
        actions = pad
    return np.asarray(actions, dtype=np.float32), {
        "valid_length": len(expert_actions),
        "policy_padded_length": int(len(pad)),
        "padded_length": max(0, action_horizon - len(actions)),
        "pad_source": pad_source,
    }


def build_openpi_observation(
    images: Mapping[str, np.ndarray],
    state: Any,
    prompt: str,
) -> dict[str, Any]:
    missing = [key for key in CAMERA_KEYS if key not in images]
    if missing:
        raise ValueError(f"missing ARX camera observations: {missing}")
    obs: dict[str, Any] = {
        key: np.asarray(images[key], dtype=np.uint8) for key in CAMERA_KEYS
    }
    obs["state"] = validate_arx_state(state)
    obs["prompt"] = str(prompt)
    return obs


@dataclass(slots=True)
class OpenPIActionTransformAdapter:
    """Use the trained Policy transform chain for expert action conversion."""

    policy: Any
    action_horizon: int = ACTION_HORIZON
    action_dim: int = ACTION_DIM
    internal_action_dim: int = INTERNAL_ACTION_DIM

    def expert_to_internal(
        self,
        observation: Mapping[str, Any],
        expert_actions: Any,
    ) -> tuple[Any, np.ndarray]:
        # OpenPI's DeltaActions transform updates arrays in-place.  Keep the
        # recorded absolute expert trajectory and observation immutable.
        actions = np.array(expert_actions, dtype=np.float32, copy=True)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"expert actions must be (T,20), got {actions.shape}")
        if len(actions) == 0:
            raise ValueError("expert actions cannot be empty")
        if len(actions) < self.action_horizon:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)],
                axis=0,
            )
        actions = actions[: self.action_horizon]
        payload = copy.deepcopy(dict(observation))
        payload["actions"] = actions
        transformed = self.policy._input_transform(payload)
        internal = np.asarray(transformed["actions"], dtype=np.float32)
        if internal.shape != (self.action_horizon, self.internal_action_dim):
            raise ValueError(
                "OpenPI input transform produced unexpected action shape "
                f"{internal.shape}; expected ({self.action_horizon},32)"
            )
        model_obs = {key: value for key, value in transformed.items() if key != "actions"}
        return model_obs, internal

    def internal_to_env(
        self,
        transformed_observation: Mapping[str, Any],
        internal_actions: Any,
    ) -> np.ndarray:
        payload = copy.deepcopy(dict(transformed_observation))
        payload["actions"] = np.array(internal_actions, dtype=np.float32, copy=True)
        output = self.policy._output_transform(payload)
        actions = np.asarray(output["actions"], dtype=np.float32)
        return actions[:, : self.action_dim]


def extract_vlm_feature(policy: Any, observation: Mapping[str, Any]) -> np.ndarray:
    """Return the fused three-camera/language pi0.5 prefix representation."""
    hidden_state, prefix_mask = policy.get_prefix_rep(dict(observation))
    mask = np.asarray(prefix_mask, dtype=np.float32)[..., None]
    denominator = np.maximum(mask.sum(axis=1), 1.0)
    pooled = jax.lax.stop_gradient((hidden_state * mask).sum(axis=1) / denominator)
    feature = np.asarray(pooled[0], dtype=np.float32)
    if feature.ndim != 1:
        raise ValueError(f"unexpected VLM feature shape: {feature.shape}")
    return feature
