"""Episode-boundary ARX FlowDAgger training runtime."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze

from arx_adapter import (
    ACTION_DIM,
    ACTION_HORIZON,
    INTERNAL_ACTION_DIM,
    MIN_INTERVENTION_VALID_LENGTH,
    OpenPIActionTransformAdapter,
    build_openpi_observation,
    extract_vlm_feature,
    pad_actions_with_policy,
    policy_rows_without_later_intervention,
    predicted_actions_from_row,
    resolve_base_model_identity,
    trim_boundary_holds,
)
from arx_campaign import get_campaign_config
from arx_demo_buffer import iter_demo_episode_dirs, iter_raw_expert_windows
from arx_inversion_cache import (
    CachedInversion,
    inversion_cache_dir,
    load_episode_cache,
    save_episode_cache,
    split_windows_by_cache,
    window_cache_key,
)
from arx_replay import collect_online_sources
from flow_matching_inverter import FlowMatchingInverter
from steering_policy import SteeringPolicy
from train_utils import _expand_noise_basis, _project_noise_to_basis

_BC_MAX_BATCH_SIZE = 64
_BC_EPOCHS = 50
_BC_MIN_MILESTONES = 5
MINIMUM_VALIDATION_REDUCTION = 0.15
_cfg = get_campaign_config()
ONLINE_BC_STEPS = _cfg.online_bc_steps
ONLINE_BC_BATCH_SIZE = _cfg.online_bc_batch_size
ONLINE_INTERVENTION_MIX = _cfg.online_intervention_mix
ONLINE_AUTONOMOUS_MIX = _cfg.online_autonomous_mix
ONLINE_DEMONSTRATION_MIX = _cfg.online_demonstration_mix
NORM_FREEZE_MIN_SUCCESS_EPISODES = _cfg.norm_freeze_min_success_episodes


def online_batch_mix_sizes(
    batch_size: int,
    n_intervention: int = 0,
    n_autonomous: int = 0,
    n_demonstration: int = 0,
) -> tuple[int, int, int]:
    """Split one BC batch 4:4:2 across intervention/autonomous/demonstration."""
    batch_size = int(batch_size)
    available = {
        "intervention": max(int(n_intervention), 0),
        "autonomous": max(int(n_autonomous), 0),
        "demonstration": max(int(n_demonstration), 0),
    }
    cfg = get_campaign_config()
    weights = {
        "intervention": cfg.online_intervention_mix,
        "autonomous": cfg.online_autonomous_mix,
        "demonstration": cfg.online_demonstration_mix,
    }
    active = [name for name, count in available.items() if count > 0]
    if batch_size < 1 or not active:
        return 0, 0, 0
    if batch_size < len(active):
        chosen = active[:batch_size]
        return (
            int("intervention" in chosen),
            int("autonomous" in chosen),
            int("demonstration" in chosen),
        )
    total_weight = sum(weights[name] for name in active)
    counts = {name: 0 for name in weights}
    remaining = batch_size
    for index, name in enumerate(active):
        leftover_sources = len(active) - index - 1
        if leftover_sources == 0:
            counts[name] = remaining
            break
        size = int(round(batch_size * weights[name] / total_weight))
        size = min(max(size, 1), remaining - leftover_sources)
        counts[name] = size
        remaining -= size
    return (
        int(counts["intervention"]),
        int(counts["autonomous"]),
        int(counts["demonstration"]),
    )


def online_episode_isolated_split(origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the entire current episode; train on history plus demonstrations."""
    origins = np.asarray(origins)
    indices = np.arange(len(origins), dtype=np.int64)
    current = origins == "current"
    return indices[~current], indices[current]


def should_freeze_target_normalization(
    n_success_episodes: int, already_frozen: bool
) -> bool:
    """Freeze z-score stats after enough distinct success episodes, not the first."""
    if already_frozen:
        return True
    return int(n_success_episodes) >= get_campaign_config().norm_freeze_min_success_episodes


def summarize_reconstruction_errors(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _format_error_mean(summary: Mapping[str, Any]) -> str:
    mean = summary.get("mean")
    if mean is None:
        return "n/a"
    return f"{float(mean):.3e}"


def action_reconstruction_mse(reconstructed: Any, target: Any) -> np.ndarray:
    reconstructed = jnp.asarray(reconstructed)
    target = jnp.asarray(target)
    return np.asarray(
        jnp.mean(
            jnp.square(
                reconstructed[..., :ACTION_DIM] - target[..., :ACTION_DIM]
            ),
            axis=(-2, -1),
        )
    )


def schedule_bc_hyperparams(n_train: int) -> tuple[int, int]:
    """Size BC from the train split: batch=min(64, n_train), 50 epochs."""
    n_train = int(n_train)
    if n_train < 1:
        raise ValueError("n_train must be positive")
    bc_batch_size = min(_BC_MAX_BATCH_SIZE, n_train)
    steps_per_epoch = (n_train + bc_batch_size - 1) // bc_batch_size
    return bc_batch_size, _BC_EPOCHS * steps_per_epoch


def bc_milestone_steps(bc_steps: int, min_points: int = _BC_MIN_MILESTONES) -> list[int]:
    """Evenly spaced validation checkpoints, including step 0 and the last step."""
    bc_steps = int(bc_steps)
    if bc_steps < 1:
        return [0]
    n_points = min(max(int(min_points), 2), bc_steps + 1)
    steps = {
        int(round(index * bc_steps / (n_points - 1)))
        for index in range(n_points)
    }
    steps.add(0)
    steps.add(bc_steps)
    return sorted(steps)


class ARXFlowDaggerRuntime:
    """Shared base model, steering actor, inverter and atomic policy versions."""

    def __init__(
        self,
        *,
        openpi_root: str,
        config_name: str,
        checkpoint_dir: str,
        output_root: str,
        default_prompt: str,
        noise_basis_k: int = ACTION_HORIZON,
        inversion_batch_size: int | None = None,
        inversion_mse_threshold: float | None = None,
        bc_steps: int = 100,
        bc_batch_size: int = 256,
        steering_lr: float | None = None,
        seed: int = 42,
    ) -> None:
        self.runtime_mode = "flowdagger"
        cfg = get_campaign_config()
        if inversion_batch_size is None:
            inversion_batch_size = cfg.inversion_batch_size
        if inversion_mse_threshold is None:
            inversion_mse_threshold = cfg.inversion_mse_threshold
        if steering_lr is None:
            steering_lr = cfg.steering_lr
        if not 1 <= inversion_batch_size <= 4:
            raise ValueError("inversion_batch_size must be in [1,4]")
        root = Path(openpi_root)
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root / "packages" / "openpi-client" / "src"))
        from openpi.policies import policy_config
        from openpi.training import config

        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.default_prompt = default_prompt
        self.config_name = config_name
        self.base_identity = resolve_base_model_identity(checkpoint_dir)
        self.base_model_id = self.base_identity["base_model_id"]
        self.steering_eligible = False
        self.noise_basis_k = int(noise_basis_k)
        self.inversion_batch_size = int(inversion_batch_size)
        self.inversion_mse_threshold = float(inversion_mse_threshold)
        # Overwritten at train time: batch=min(64, n_train), steps=50 epochs.
        self.bc_steps = int(bc_steps)
        self.bc_batch_size = int(bc_batch_size)
        self.steering_lr = float(steering_lr)
        if not np.isfinite(self.steering_lr) or self.steering_lr <= 0:
            raise ValueError("steering_lr must be finite and positive")
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        train_config = config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
        )
        raw_model = getattr(self.policy, "_model", None)
        if raw_model is None:
            raise RuntimeError("OpenPI Policy does not expose its shared _model")
        if raw_model.action_horizon != ACTION_HORIZON or raw_model.action_dim != INTERNAL_ACTION_DIM:
            raise ValueError(
                f"checkpoint model mismatch: horizon={raw_model.action_horizon}, "
                f"action_dim={raw_model.action_dim}"
            )
        self.action_adapter = OpenPIActionTransformAdapter(self.policy)
        self.inverter = FlowMatchingInverter(
            raw_model,
            method="perstep_fp",
            num_denoise_steps=10,
            fp_per_step=5,
            seed=seed,
        )
        self.steering: SteeringPolicy | None = None
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None
        self.target_clip_low: np.ndarray | None = None
        self.target_clip_high: np.ndarray | None = None
        self._normalization_frozen = False
        self.policy_version = 0
        self._online_context: dict[str, Any] | None = None
        self.bootstrap_reviewed_episodes = 0
        self.campaign_id = cfg.campaign_id
        self.demo_buffer_dir = Path(cfg.demo_buffer_dir)
        self.window_stride = int(cfg.window_stride)
        self._restore_active_version()

    def _predict_coefficients(self, actor_obs: dict[str, Any]) -> np.ndarray:
        """Predict, deployment-clip, and de-normalize noise coefficients."""
        if self.steering is None:
            raise RuntimeError("steering policy is not initialized")
        statistics = (
            self.target_mean,
            self.target_std,
            self.target_clip_low,
            self.target_clip_high,
        )
        if any(value is None for value in statistics):
            raise RuntimeError("steering target normalization is not initialized")
        normalized = self.steering.sample_actions(actor_obs).reshape(
            self.noise_basis_k, INTERNAL_ACTION_DIM
        )
        normalized = np.clip(
            normalized, self.target_clip_low, self.target_clip_high
        )
        return normalized * self.target_std + self.target_mean

    def infer(self, observation: dict[str, Any], *, shadow_mode: bool) -> dict[str, Any]:
        with self._lock:
            base_result = None
            if self.steering is None or shadow_mode:
                base_result = self.policy.infer(observation)
            if self.steering is None:
                return {
                    "actions": np.asarray(base_result["actions"], dtype=np.float32),
                    "model_infer_ms": float(base_result.get("policy_timing", {}).get("infer_ms", 0.0)),
                    "executed_policy": "base",
                }

            feature = extract_vlm_feature(self.policy, observation)
            state = np.asarray(observation["state"], dtype=np.float32)
            actor_obs = {
                "pixels": feature[None, :, None],
                "state": state[None, :, None],
            }
            coefficients = self._predict_coefficients(actor_obs)
            noise = _expand_noise_basis(coefficients, action_horizon=ACTION_HORIZON)
            steering_result = self.policy.infer(observation, noise=noise)
            if shadow_mode:
                return {
                    "actions": np.asarray(base_result["actions"], dtype=np.float32),
                    "shadow_actions": np.asarray(steering_result["actions"], dtype=np.float32),
                    "model_infer_ms": float(base_result.get("policy_timing", {}).get("infer_ms", 0.0)),
                    "executed_policy": "base_shadow",
                }
            return {
                "actions": np.asarray(steering_result["actions"], dtype=np.float32),
                "model_infer_ms": float(steering_result.get("policy_timing", {}).get("infer_ms", 0.0)),
                "executed_policy": "steering",
            }

    def train_episode(self, episode_dir: Path, current_version: int) -> dict[str, Any]:
        sources = collect_online_sources(self.output_root, Path(episode_dir))
        intervention_dirs = list(sources["current"]) + list(
            sources["intervention_history"]
        )
        self._online_context = {
            "current": str(Path(episode_dir).resolve()),
            "autonomous_dirs": [
                str(Path(path).resolve()) for path in sources["autonomous"]
            ],
            "demonstration_dir": str(self.demo_buffer_dir),
        }
        try:
            return self.train_episodes(
                intervention_dirs, current_version, metrics_dir=Path(episode_dir)
            )
        finally:
            self._online_context = None

    def train_episodes(
        self,
        episode_dirs: Iterable[Path],
        current_version: int,
        *,
        metrics_dir: Path,
        min_validation_reduction: float | None = None,
    ) -> dict[str, Any]:
        """Train transactionally; any failure restores the in-memory actor."""
        with self._lock:
            had_steering = self.steering is not None
            previous_actor = self.steering._actor if had_steering else None
            previous_rng = self.steering._rng if had_steering else None
            previous_normalization = tuple(
                None if value is None else value.copy()
                for value in (
                    self.target_mean,
                    self.target_std,
                    self.target_clip_low,
                    self.target_clip_high,
                )
            )
            previous_frozen = self._normalization_frozen
            try:
                return self._train_episodes_locked(
                    list(episode_dirs),
                    current_version,
                    metrics_dir=metrics_dir,
                    min_validation_reduction=min_validation_reduction,
                )
            except Exception:
                if not had_steering:
                    self.steering = None
                else:
                    self.steering._actor = previous_actor
                    self.steering._rng = previous_rng
                (
                    self.target_mean,
                    self.target_std,
                    self.target_clip_low,
                    self.target_clip_high,
                ) = previous_normalization
                self._normalization_frozen = previous_frozen
                raise

    def _train_episodes_locked(
        self,
        episode_dirs: list[Path],
        current_version: int,
        *,
        metrics_dir: Path,
        min_validation_reduction: float | None,
    ) -> dict[str, Any]:
        with self._lock:
            training_started = time.monotonic()
            def progress(message: str) -> None:
                elapsed = time.monotonic() - training_started
                print(f"[FlowDAgger][{elapsed:8.1f}s] {message}", flush=True)

            if current_version != self.policy_version:
                raise RuntimeError(
                    f"stale training request: current={current_version}, active={self.policy_version}"
                )
            windows = []
            current_key = (
                str(self._online_context.get("current", ""))
                if self._online_context is not None
                else ""
            )
            for episode_dir in episode_dirs:
                origin = (
                    "current"
                    if self._online_context is not None
                    and str(Path(episode_dir).resolve()) == current_key
                    else "history"
                    if self._online_context is not None
                    else "offline"
                )
                for observation, actions, info in self._load_expert_windows(episode_dir):
                    info["source"] = "intervention"
                    info["origin"] = origin
                    info["episode_path"] = str(Path(episode_dir).resolve())
                    windows.append((observation, actions, info))
            if self._online_context is not None:
                for episode_path in self._online_context.get("autonomous_dirs", []):
                    episode_dir = Path(episode_path)
                    origin = (
                        "current"
                        if str(episode_dir.resolve()) == current_key
                        else "history"
                    )
                    for observation, actions, info in self._load_autonomous_windows(
                        episode_dir
                    ):
                        info["source"] = "autonomous"
                        info["origin"] = origin
                        info["episode_path"] = str(episode_dir.resolve())
                        windows.append((observation, actions, info))
                demo_root = Path(
                    self._online_context.get("demonstration_dir", self.demo_buffer_dir)
                )
                for episode_dir in iter_demo_episode_dirs(demo_root):
                    for observation, actions, info in iter_raw_expert_windows(
                        episode_dir, prompt=self.default_prompt
                    ):
                        info["episode_path"] = str(Path(episode_dir).absolute())
                        windows.append((observation, actions, info))
            if not windows:
                raise ValueError("successful episode contains no complete expert transitions")
            progress(
                f"阶段 1/4 数据准备: episodes={len(episode_dirs)} windows={len(windows)}"
            )

            samples: list[
                tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]
            ] = []
            inversion_report: list[dict[str, Any]] = []
            accepted_model_obs: list[Any] = []
            accepted_targets: list[np.ndarray] = []
            episode_caches: dict[str, dict[str, CachedInversion]] = {}
            dirty_episodes: set[str] = set()
            demo_root = self._demo_root_for_cache()

            def records_for_episode(episode_path: str) -> dict[str, CachedInversion]:
                if episode_path not in episode_caches:
                    episode_caches[episode_path] = load_episode_cache(
                        inversion_cache_dir(episode_path, demo_root=demo_root),
                        base_model_id=self.base_model_id,
                        noise_basis_k=self.noise_basis_k,
                        inversion_mse_threshold=self.inversion_mse_threshold,
                    )
                return episode_caches[episode_path]

            def flush_dirty_caches() -> None:
                for episode_path in list(dirty_episodes):
                    save_episode_cache(
                        inversion_cache_dir(episode_path, demo_root=demo_root),
                        episode_caches[episode_path],
                        base_model_id=self.base_model_id,
                        noise_basis_k=self.noise_basis_k,
                        inversion_mse_threshold=self.inversion_mse_threshold,
                    )
                    dirty_episodes.discard(episode_path)

            cached_samples, missed_windows, cache_reports = split_windows_by_cache(
                windows, records_for_episode
            )
            samples.extend(cached_samples)
            inversion_report.extend(cache_reports)
            n_cache_hits = len(windows) - len(missed_windows)
            n_cache_misses = len(missed_windows)

            prepared = []
            from openpi.models import model as model_module
            for observation, expert_actions, window_info in missed_windows:
                transformed_obs, target = self.action_adapter.expert_to_internal(
                    observation, expert_actions
                )
                batched_obs = jax.tree.map(
                    lambda value: jnp.asarray(value)[None, ...], transformed_obs
                )
                model_obs = model_module.Observation.from_dict(batched_obs)
                prepared.append(
                    (observation, model_obs, jnp.asarray(target)[None, ...], window_info)
                )

            inversion_started = time.monotonic()
            total_batches = (
                (len(prepared) + self.inversion_batch_size - 1)
                // self.inversion_batch_size
                if prepared else 0
            )
            progress(
                f"阶段 2/4 BC 反演开始: windows={len(windows)} "
                f"cache_hits={n_cache_hits} invert={len(prepared)} "
                f"batches={total_batches} batch_size={self.inversion_batch_size} "
                f"noise_basis_k={self.noise_basis_k}"
            )
            try:
                for batch_start in range(0, len(prepared), self.inversion_batch_size):
                    batch_items = prepared[batch_start : batch_start + self.inversion_batch_size]
                    model_obs = jax.tree.map(
                        lambda *values: jnp.concatenate(values, axis=0),
                        *[item[1] for item in batch_items],
                    )
                    target_batch = jnp.concatenate(
                        [item[2] for item in batch_items], axis=0
                    )
                    noise, _ = self.inverter.invert(model_obs, target_batch)
                    reconstructed = self.inverter._denoise(model_obs, noise)
                    batch_mse = action_reconstruction_mse(reconstructed, target_batch)
                    repr_noise = []
                    for item_index in range(len(batch_items)):
                        noise_np = np.asarray(noise[item_index], dtype=np.float32)
                        coefficients = _project_noise_to_basis(
                            noise_np, self.noise_basis_k
                        )
                        expanded = _expand_noise_basis(
                            coefficients, action_horizon=ACTION_HORIZON
                        )
                        repr_noise.append(np.asarray(expanded[0], dtype=np.float32))
                    repr_noise_batch = jnp.asarray(np.stack(repr_noise, axis=0))
                    if self.noise_basis_k == ACTION_HORIZON:
                        repr_mse = batch_mse
                    else:
                        repr_reconstructed = self.inverter._denoise(
                            model_obs, repr_noise_batch
                        )
                        repr_mse = action_reconstruction_mse(
                            repr_reconstructed, target_batch
                        )
                    for item_index, (
                        observation, item_obs, item_target, window_info
                    ) in enumerate(batch_items):
                        active_mse = float(batch_mse[item_index])
                        e_repr = float(repr_mse[item_index])
                        accepted = bool(
                            np.isfinite(active_mse)
                            and active_mse <= self.inversion_mse_threshold
                        )
                        info = dict(window_info)
                        info["from_cache"] = False
                        inversion_report.append({
                            **info,
                            "normalized_action_mse": active_mse,
                            "e_full": active_mse,
                            "e_repr": e_repr,
                            "e_actor_before": None,
                            "e_actor_after": None,
                            "accepted": accepted,
                            "from_cache": False,
                        })
                        feature = None
                        coefficients = None
                        if accepted:
                            noise_np = np.asarray(noise[item_index], dtype=np.float32)
                            coefficients = _project_noise_to_basis(
                                noise_np, self.noise_basis_k
                            )
                            feature = extract_vlm_feature(self.policy, observation)
                            samples.append((
                                feature,
                                np.asarray(observation["state"], dtype=np.float32),
                                coefficients,
                                active_mse,
                                info,
                            ))
                            accepted_model_obs.append(item_obs)
                            accepted_targets.append(np.asarray(item_target[0]))
                            inversion_report[-1]["sample_index"] = len(samples) - 1
                        episode_path = str(window_info["episode_path"])
                        records_for_episode(episode_path)[window_cache_key(window_info)] = (
                            CachedInversion(
                                accepted=accepted,
                                mse=active_mse,
                                e_repr=e_repr,
                                feature=feature,
                                state=(
                                    np.asarray(observation["state"], dtype=np.float32)
                                    if accepted else None
                                ),
                                coefficients=coefficients,
                            )
                        )
                        dirty_episodes.add(episode_path)
                    flush_dirty_caches()
                    completed_batches = batch_start // self.inversion_batch_size + 1
                    elapsed = time.monotonic() - inversion_started
                    eta = (
                        elapsed / completed_batches * (total_batches - completed_batches)
                    )
                    accepted_count = sum(item["accepted"] for item in inversion_report)
                    processed_count = len(inversion_report)
                    progress(
                        f"BC 反演 {completed_batches}/{total_batches} "
                        f"({completed_batches / total_batches:.0%}) "
                        f"accepted={accepted_count}/{processed_count} "
                        f"cache_hits={n_cache_hits} ETA={eta:.0f}s"
                    )
            finally:
                flush_dirty_caches()
            metrics_dir.mkdir(parents=True, exist_ok=True)
            if not samples:
                self._atomic_json(metrics_dir / "inversion_report.json", {
                    "threshold": self.inversion_mse_threshold,
                    "cache_hits": n_cache_hits,
                    "cache_misses": n_cache_misses,
                    "chunks": inversion_report,
                })
                raise ValueError("all inversion chunks exceeded the MSE threshold")

            if self.steering is not None and accepted_model_obs:
                progress("阶段 2/4 计算 E_actor_before")
                actor_before = self._actor_reconstruction_mses(
                    accepted_model_obs,
                    accepted_targets,
                    np.stack([
                        item[0] for item in samples if not item[4].get("from_cache")
                    ]),
                    np.stack([
                        item[1] for item in samples if not item[4].get("from_cache")
                    ]),
                )
                sample_index = 0
                for row in inversion_report:
                    if not row.get("accepted") or row.get("from_cache"):
                        continue
                    row["e_actor_before"] = float(actor_before[sample_index])
                    sample_index += 1

            self._atomic_json(metrics_dir / "inversion_report.json", {
                "threshold": self.inversion_mse_threshold,
                "cache_hits": n_cache_hits,
                "cache_misses": n_cache_misses,
                "chunks": inversion_report,
            })
            e_full_summary = summarize_reconstruction_errors(
                np.array([item[3] for item in samples])
            )
            e_repr_summary = summarize_reconstruction_errors(
                np.array([
                    row["e_repr"] for row in inversion_report if row.get("accepted")
                ])
            )
            progress(
                f"阶段 2/4 BC 反演完成: accepted={len(samples)}/{len(inversion_report)} "
                f"acceptance_rate={len(samples) / len(inversion_report):.1%} "
                f"cache_hits={n_cache_hits} inverted={n_cache_misses} "
                f"e_full={_format_error_mean(e_full_summary)} "
                f"e_repr={_format_error_mean(e_repr_summary)}"
            )

            features = np.stack([item[0] for item in samples])
            states = np.stack([item[1] for item in samples])
            raw_targets = np.stack([item[2] for item in samples])

            indices = np.arange(len(samples), dtype=np.int64)
            episode_ids = np.asarray([item[4]["episode_id"] for item in samples])
            padded_lengths = np.asarray(
                [int(item[4]["padded_length"]) for item in samples], dtype=np.int64
            )
            unique_episodes = np.asarray(sorted(set(episode_ids.tolist())))
            sample_sources = np.asarray(
                [item[4].get("source", "intervention") for item in samples]
            )
            sample_origins = np.asarray(
                [item[4].get("origin", "offline") for item in samples]
            )
            online = self._online_context is not None
            intervention_indices = indices[sample_sources == "intervention"] if online else indices
            autonomous_indices = indices[sample_sources == "autonomous"] if online else indices[:0]
            demonstration_indices = (
                indices[sample_sources == "demonstration"] if online else indices[:0]
            )
            current_indices = indices[
                (sample_sources == "intervention") & (sample_origins == "current")
            ] if online else indices[:0]
            if min_validation_reduction is not None and len(indices) < 5:
                raise RuntimeError(
                    "offline validation gate requires at least 5 accepted chunks"
                )

            # Offline validation must isolate complete episodes.  Randomly
            # splitting overlapping stride-10 windows leaks nearly identical
            # observations and future actions across train and validation.
            # Online does the same for the current episode: it is validation
            # only, and enters the train mix on the next successful update.
            if online:
                if len(current_indices) == 0:
                    return {
                        "state": "no_improvement",
                        "reason": "no_current_windows",
                        "policy_version": current_version,
                    }
                train_indices, validation_indices = online_episode_isolated_split(
                    sample_origins
                )
                train_episodes = np.unique(episode_ids[train_indices])
                validation_episodes = np.unique(episode_ids[validation_indices])
            elif min_validation_reduction is not None:
                if len(unique_episodes) < 2:
                    raise RuntimeError(
                        "episode-isolated validation requires at least 2 episodes"
                    )
                shuffled_episodes = unique_episodes.copy()
                self._rng.shuffle(shuffled_episodes)
                validation_episode_count = max(
                    1, int(np.ceil(len(shuffled_episodes) * 0.2))
                )
                validation_episodes = shuffled_episodes[-validation_episode_count:]
                train_episodes = shuffled_episodes[:-validation_episode_count]
                train_indices = indices[np.isin(episode_ids, train_episodes)]
                validation_indices = indices[np.isin(episode_ids, validation_episodes)]
            else:
                shuffled_indices = indices.copy()
                self._rng.shuffle(shuffled_indices)
                split = max(1, int(np.floor(len(shuffled_indices) * 0.8)))
                if len(shuffled_indices) > 1:
                    split = min(split, len(shuffled_indices) - 1)
                train_indices = shuffled_indices[:split]
                validation_indices = (
                    shuffled_indices[split:]
                    if split < len(shuffled_indices)
                    else shuffled_indices[-1:]
                )
                train_episodes = np.unique(episode_ids[train_indices])
                validation_episodes = np.unique(episode_ids[validation_indices])

            if len(train_indices) == 0 or len(validation_indices) == 0:
                raise RuntimeError("empty train or validation split")

            if online:
                online_cfg = get_campaign_config()
                self.bc_batch_size = online_cfg.online_bc_batch_size
                self.bc_steps = online_cfg.online_bc_steps
            else:
                self.bc_batch_size, self.bc_steps = schedule_bc_hyperparams(
                    len(train_indices)
                )

            n_success_episodes = int(len(np.unique(
                episode_ids[train_indices][sample_sources[train_indices] != "demonstration"]
            ))) if online else int(len(np.unique(episode_ids[train_indices])))
            fitted_this_round = False
            if not self._normalization_frozen:
                self._fit_target_normalization(raw_targets[train_indices])
                fitted_this_round = True
                self._normalization_frozen = should_freeze_target_normalization(
                    n_success_episodes, False
                )
                progress(
                    f"目标归一化已拟合: success_episodes={n_success_episodes} "
                    f"frozen={self._normalization_frozen} "
                    f"min_freeze={get_campaign_config().norm_freeze_min_success_episodes}"
                )
            if any(value is None for value in (
                self.target_mean,
                self.target_std,
                self.target_clip_low,
                self.target_clip_high,
            )):
                raise RuntimeError("target normalization is incomplete")
            targets = np.clip(
                (raw_targets - self.target_mean) / self.target_std,
                self.target_clip_low,
                self.target_clip_high,
            )

            if self.steering is None:
                self.steering = SteeringPolicy(
                    seed=42,
                    observations={
                        "pixels": features[:1, :, None],
                        "state": states[:1, :, None],
                    },
                    actions=targets[:1],
                    lr=self.steering_lr,
                    hidden_dims=(256, 256, 256),
                    latent_dim=256,
                    encoder_type="vlm_pi0",
                    color_jitter=False,
                    action_magnitude=3.0,
                    num_cameras=3,
                    output_bound="tanh",
                )

            def subset_loss(subset: np.ndarray) -> float | None:
                if len(subset) == 0:
                    return None
                return self.steering.evaluate_mse(
                    self._batch(features, states, targets, subset)
                )

            train_full = train_indices[padded_lengths[train_indices] == 0]
            train_padded = train_indices[padded_lengths[train_indices] > 0]
            validation_full = validation_indices[
                padded_lengths[validation_indices] == 0
            ]
            validation_padded = validation_indices[
                padded_lengths[validation_indices] > 0
            ]

            target_abs_over_3 = np.abs(raw_targets) > 3.0
            outlier_counts = np.count_nonzero(target_abs_over_3, axis=0)
            outlier_dimensions = []
            for basis_index, action_index in np.argwhere(outlier_counts > 0):
                values = raw_targets[:, basis_index, action_index]
                outlier_dimensions.append({
                    "basis_index": int(basis_index),
                    "action_index": int(action_index),
                    "count": int(outlier_counts[basis_index, action_index]),
                    "fraction": float(np.mean(np.abs(values) > 3.0)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                })
            outlier_windows = []
            per_window_counts = np.count_nonzero(target_abs_over_3, axis=(1, 2))
            for sample_index in np.flatnonzero(per_window_counts):
                locations = np.argwhere(target_abs_over_3[sample_index])
                info = samples[int(sample_index)][4]
                outlier_windows.append({
                    "sample_index": int(sample_index),
                    "episode_id": str(info["episode_id"]),
                    "start_step_id": int(info["start_step_id"]),
                    "valid_length": int(info["valid_length"]),
                    "padded_length": int(info["padded_length"]),
                    "outlier_count": int(per_window_counts[sample_index]),
                    "max_abs": float(np.max(np.abs(raw_targets[sample_index]))),
                    "dimensions": [
                        {"basis_index": int(row[0]), "action_index": int(row[1])}
                        for row in locations
                    ],
                })
            self._atomic_json(metrics_dir / "target_outlier_report.json", {
                "threshold_abs": 3.0,
                "total_count": int(np.count_nonzero(target_abs_over_3)),
                "total_fraction": float(np.mean(target_abs_over_3)),
                "dimensions": sorted(
                    outlier_dimensions, key=lambda item: item["count"], reverse=True
                ),
                "windows": sorted(
                    outlier_windows, key=lambda item: item["max_abs"], reverse=True
                ),
            })
            normalized_outside_clip = np.logical_or(
                targets < self.target_clip_low,
                targets > self.target_clip_high,
            )
            target_statistics = {
                "shape": list(raw_targets.shape),
                "mean": float(np.mean(raw_targets)),
                "std": float(np.std(raw_targets)),
                "min": float(np.min(raw_targets)),
                "max": float(np.max(raw_targets)),
                "abs_over_3_count": int(np.count_nonzero(target_abs_over_3)),
                "abs_over_3_fraction": float(np.mean(target_abs_over_3)),
                "normalized_mean": float(np.mean(targets)),
                "normalized_std": float(np.std(targets)),
                "normalized_min": float(np.min(targets)),
                "normalized_max": float(np.max(targets)),
                "normalized_outside_deployment_clip_count": int(
                    np.count_nonzero(normalized_outside_clip)
                ),
                "normalized_outside_deployment_clip_fraction": float(
                    np.mean(normalized_outside_clip)
                ),
                "normalization_std_min": float(np.min(self.target_std)),
                "normalization_std_max": float(np.max(self.target_std)),
                "clip_low_min": float(np.min(self.target_clip_low)),
                "clip_high_max": float(np.max(self.target_clip_high)),
            }

            milestone_steps = bc_milestone_steps(self.bc_steps)
            milestone_set = set(milestone_steps)
            loss_curve: list[dict[str, Any]] = []

            def record_losses(step: int) -> None:
                train_loss = subset_loss(train_indices)
                validation_loss = subset_loss(validation_indices)
                validation_initial = loss_curve[0]["validation_loss"] if loss_curve else validation_loss
                reduction = (
                    1.0 - validation_loss / validation_initial
                    if validation_loss is not None
                    and validation_initial is not None
                    and validation_initial > 0
                    else 0.0
                )
                loss_curve.append({
                    "step": int(step),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_reduction": float(reduction),
                    "train_full_loss": subset_loss(train_full),
                    "train_padded_loss": subset_loss(train_padded),
                    "validation_full_loss": subset_loss(validation_full),
                    "validation_padded_loss": subset_loss(validation_padded),
                })

            record_losses(0)
            mix_intervention = 0
            mix_autonomous = 0
            mix_demonstration = 0
            if online:
                mix_intervention, mix_autonomous, mix_demonstration = online_batch_mix_sizes(
                    self.bc_batch_size,
                    n_intervention=int(np.count_nonzero(
                        np.isin(intervention_indices, train_indices)
                    )),
                    n_autonomous=int(np.count_nonzero(
                        np.isin(autonomous_indices, train_indices)
                    )),
                    n_demonstration=int(np.count_nonzero(
                        np.isin(demonstration_indices, train_indices)
                    )),
                )
            progress(
                f"阶段 3/4 Steering BC 开始: steps={self.bc_steps} "
                f"batch_size={self.bc_batch_size} "
                f"lr={self.steering_lr:g} "
                f"mix={mix_intervention}/{mix_autonomous}/{mix_demonstration} "
                f"train_windows={len(train_indices)} validation_windows={len(validation_indices)} "
                f"milestones={milestone_steps} "
                f"initial_validation_loss={loss_curve[0]['validation_loss']:.6g}"
            )
            best_step = 0
            best_validation_loss = float(loss_curve[0]["validation_loss"])
            best_actor = self.steering._actor
            best_rng = self.steering._rng
            train_losses = []
            bc_started = time.monotonic()
            progress_interval = max(1, min(50, self.bc_steps // 20 or 1))
            for step in range(1, self.bc_steps + 1):
                mixed_sources = (
                    mix_intervention,
                    mix_autonomous,
                    mix_demonstration,
                )
                if online and sum(count > 0 for count in mixed_sources) >= 2:
                    parts = []
                    for pool, count in (
                        (intervention_indices, mix_intervention),
                        (autonomous_indices, mix_autonomous),
                        (demonstration_indices, mix_demonstration),
                    ):
                        if count <= 0 or len(pool) == 0:
                            continue
                        parts.append(
                            self._rng.choice(
                                pool,
                                size=count,
                                replace=len(pool) < count,
                            )
                        )
                    batch_indices = np.concatenate(parts)
                    self._rng.shuffle(batch_indices)
                else:
                    batch_indices = self._rng.choice(
                        train_indices,
                        size=self.bc_batch_size,
                        replace=len(train_indices) < self.bc_batch_size,
                    )
                info = self.steering.update(self._batch(features, states, targets, batch_indices))
                train_losses.append(float(info["bc_loss"]))
                if step in milestone_set:
                    record_losses(step)
                    current_validation_loss = float(
                        loss_curve[-1]["validation_loss"]
                    )
                    if current_validation_loss < best_validation_loss:
                        best_step = step
                        best_validation_loss = current_validation_loss
                        best_actor = self.steering._actor
                        best_rng = self.steering._rng
                if step % progress_interval == 0 or step == self.bc_steps:
                    elapsed = time.monotonic() - bc_started
                    eta = elapsed / step * (self.bc_steps - step)
                    recent_loss = float(np.mean(train_losses[-progress_interval:]))
                    progress(
                        f"Steering BC {step}/{self.bc_steps} ({step / self.bc_steps:.0%}) "
                        f"recent_bc_loss={recent_loss:.6g} "
                        f"best_validation_loss={best_validation_loss:.6g} ETA={eta:.0f}s"
                    )

            if loss_curve[-1]["step"] != self.bc_steps:
                record_losses(self.bc_steps)

            validation_before = float(loss_curve[0]["validation_loss"])
            if not online:
                self.steering._actor = best_actor
                self.steering._rng = best_rng
                validation_after = best_validation_loss
            else:
                best_step = self.bc_steps
                validation_after = float(loss_curve[-1]["validation_loss"])
            if not np.isfinite(validation_after):
                raise FloatingPointError("steering validation loss is NaN/Inf")
            validation_reduction = (
                1.0 - validation_after / validation_before if validation_before > 0 else 0.0
            )
            progress(
                f"阶段 3/4 Steering BC 完成: best_step={best_step} "
                f"validation_loss={validation_before:.6g}->{validation_after:.6g} "
                f"reduction={validation_reduction:.1%}"
            )
            if accepted_model_obs:
                progress("阶段 3/4 计算 E_actor_after")
                actor_after = self._actor_reconstruction_mses(
                    accepted_model_obs,
                    accepted_targets,
                    np.stack([
                        item[0] for item in samples if not item[4].get("from_cache")
                    ]),
                    np.stack([
                        item[1] for item in samples if not item[4].get("from_cache")
                    ]),
                )
                sample_index = 0
                for row in inversion_report:
                    if not row.get("accepted") or row.get("from_cache"):
                        continue
                    row["e_actor_after"] = float(actor_after[sample_index])
                    sample_index += 1
            e_full_accepted = np.array(
                [row["e_full"] for row in inversion_report if row.get("accepted")],
                dtype=np.float64,
            )
            e_repr_accepted = np.array(
                [row["e_repr"] for row in inversion_report if row.get("accepted")],
                dtype=np.float64,
            )
            reconstruction = {
                "e_full": summarize_reconstruction_errors(e_full_accepted),
                "e_repr": summarize_reconstruction_errors(e_repr_accepted),
                "e_actor_before": summarize_reconstruction_errors(
                    np.array(
                        [
                            row["e_actor_before"]
                            for row in inversion_report
                            if row.get("accepted") and row.get("e_actor_before") is not None
                        ],
                        dtype=np.float64,
                    )
                ),
                "e_actor_after": summarize_reconstruction_errors(
                    np.array(
                        [
                            row["e_actor_after"]
                            for row in inversion_report
                            if row.get("accepted") and row.get("e_actor_after") is not None
                        ],
                        dtype=np.float64,
                    )
                ),
            }
            self._atomic_json(metrics_dir / "inversion_report.json", {
                "threshold": self.inversion_mse_threshold,
                "cache_hits": n_cache_hits,
                "cache_misses": n_cache_misses,
                "chunks": inversion_report,
                "reconstruction": reconstruction,
            })
            progress(
                f"重建误差: E_full={_format_error_mean(reconstruction['e_full'])} "
                f"E_repr={_format_error_mean(reconstruction['e_repr'])} "
                f"E_actor={_format_error_mean(reconstruction['e_actor_after'])}"
            )
            diagnostics = {
                "bc_steps": self.bc_steps,
                "bc_batch_size": self.bc_batch_size,
                "bc_schedule": {
                    "method": (
                        "online_fixed_steps" if online else "50_epochs_min_batch_64"
                    ),
                    "n_train": int(len(train_indices)),
                    "epochs": None if online else _BC_EPOCHS,
                    "max_batch_size": _BC_MAX_BATCH_SIZE,
                    "milestone_steps": milestone_steps,
                },
                "steering_lr": self.steering_lr,
                "best_step": int(best_step),
                "best_validation_loss": float(best_validation_loss),
                "split": {
                    "method": (
                        "replay_holdout_current_episode"
                        if online
                        else "episode_isolated_80_20"
                        if min_validation_reduction is not None
                        else "window_random_80_20"
                    ),
                    "intervention_mix": (
                        mix_intervention / self.bc_batch_size if online else None
                    ),
                    "autonomous_mix": (
                        mix_autonomous / self.bc_batch_size if online else None
                    ),
                    "demonstration_mix": (
                        mix_demonstration / self.bc_batch_size if online else None
                    ),
                    "intervention_batch_size": int(mix_intervention) if online else None,
                    "autonomous_batch_size": int(mix_autonomous) if online else None,
                    "demonstration_batch_size": int(mix_demonstration) if online else None,
                    "n_intervention_windows": int(len(intervention_indices)) if online else None,
                    "n_autonomous_windows": int(len(autonomous_indices)) if online else None,
                    "n_demonstration_windows": int(len(demonstration_indices)) if online else None,
                    "n_cache_hits": int(n_cache_hits),
                    "n_cache_misses": int(n_cache_misses),
                    "n_inverted_this_round": int(n_cache_misses),
                    "train_episodes": sorted(map(str, train_episodes.tolist())),
                    "validation_episodes": sorted(
                        map(str, validation_episodes.tolist())
                    ),
                    "train_windows": int(len(train_indices)),
                    "validation_windows": int(len(validation_indices)),
                    "train_full_windows": int(len(train_full)),
                    "train_padded_windows": int(len(train_padded)),
                    "validation_full_windows": int(len(validation_full)),
                    "validation_padded_windows": int(len(validation_padded)),
                },
                "target_statistics": target_statistics,
                "target_normalization": {
                    "method": "per_dimension_zscore",
                    "fit_on": "train_episodes_until_freeze",
                    "frozen": self._normalization_frozen,
                    "fitted_this_round": fitted_this_round,
                    "success_episodes_at_fit": n_success_episodes,
                    "min_success_episodes": get_campaign_config().norm_freeze_min_success_episodes,
                    "output_bound": "tanh_x3_standardized",
                    "deployment_clip_percentiles": [0.1, 99.9],
                    "noise_basis_k": self.noise_basis_k,
                },
                "reconstruction": reconstruction,
                "loss_curve": loss_curve,
            }
            self._atomic_json(metrics_dir / "bc_diagnostics.json", diagnostics)
            if (
                min_validation_reduction is not None
                and validation_reduction < min_validation_reduction
            ):
                raise RuntimeError(
                    "validation-loss gate failed: "
                    f"reduction={validation_reduction:.3f}, "
                    f"required={min_validation_reduction:.3f}"
                )

            versions_root = self.output_root / "steering_checkpoints"
            existing_versions = [
                int(path.name.removeprefix("version_"))
                for path in versions_root.glob("version_*")
                if path.is_dir() and path.name.removeprefix("version_").isdigit()
            ] if versions_root.exists() else []
            new_version = max([current_version, *existing_versions]) + 1
            version_metadata = {
                "format_version": 2,
                "campaign_id": self.campaign_id,
                "lineage_parent": None if current_version == 0 else current_version,
                "base_model_id": self.base_model_id,
                "base_identity": self.base_identity,
                "openpi_config": self.config_name,
                "episode_dirs": [str(path) for path in episode_dirs],
                "inversion": {
                    "threshold": self.inversion_mse_threshold,
                    "num_windows": len(windows),
                    "num_accepted": len(samples),
                    "acceptance_rate": len(samples) / len(windows),
                    "mse_mean": float(np.mean([item[3] for item in samples])),
                    "noise_basis_k": self.noise_basis_k,
                    "reconstruction": reconstruction,
                },
                "training": {
                    "bc_steps": self.bc_steps,
                    "bc_batch_size": self.bc_batch_size,
                    "steering_lr": self.steering_lr,
                    "intervention_mix": (
                        mix_intervention / self.bc_batch_size if online else None
                    ),
                    "autonomous_mix": (
                        mix_autonomous / self.bc_batch_size if online else None
                    ),
                    "demonstration_mix": (
                        mix_demonstration / self.bc_batch_size if online else None
                    ),
                    "normalization_frozen": self._normalization_frozen,
                    "validation_loss_before": validation_before,
                    "validation_loss_after": validation_after,
                    "validation_loss_reduction": validation_reduction,
                },
                "deployment_status": "active",
            }
            self._save_version(new_version, version_metadata)
            metrics = {
                "policy_version": new_version,
                "num_windows": len(windows),
                "num_kept": len(samples),
                "cache_hits": int(n_cache_hits),
                "cache_misses": int(n_cache_misses),
                "inversion_mse_mean": float(np.mean([item[3] for item in samples])),
                "reconstruction": reconstruction,
                "bc_loss_mean": float(np.mean(train_losses)),
                "validation_loss_before": validation_before,
                "validation_loss_after": validation_after,
                "validation_loss_reduction": validation_reduction,
                "bc_diagnostics": diagnostics,
                "episode_dirs": [str(path) for path in episode_dirs],
                "state": "succeeded",
            }
            self._atomic_json(metrics_dir / "training_metrics.json", metrics)
            progress(
                f"阶段 4/4 checkpoint 已保存: version={new_version} "
                f"metrics={metrics_dir / 'training_metrics.json'}"
            )
            self._activate_version(new_version)
            self.policy_version = new_version
            self.steering_eligible = True
            return metrics

    def _fit_target_normalization(self, train_targets: np.ndarray) -> None:
        self.target_mean = np.mean(
            train_targets, axis=0, dtype=np.float64
        ).astype(np.float32)
        self.target_std = np.std(
            train_targets, axis=0, dtype=np.float64
        ).astype(np.float32)
        self.target_std = np.maximum(self.target_std, 1e-6)
        normalized_train = (train_targets - self.target_mean) / self.target_std
        self.target_clip_low = np.maximum(
            np.percentile(normalized_train, 0.1, axis=0), -3.0
        ).astype(np.float32)
        self.target_clip_high = np.minimum(
            np.percentile(normalized_train, 99.9, axis=0), 3.0
        ).astype(np.float32)

    def _predict_raw_noise(self, features: np.ndarray, states: np.ndarray) -> np.ndarray:
        if self.steering is None:
            raise RuntimeError("steering policy is not initialized")
        actor_obs = {
            "pixels": features[:, :, None],
            "state": states[:, :, None],
        }
        normalized = np.asarray(
            self.steering.sample_actions(actor_obs), dtype=np.float32
        ).reshape(len(features), self.noise_basis_k, INTERNAL_ACTION_DIM)
        normalized = np.clip(
            normalized, self.target_clip_low, self.target_clip_high
        )
        coefficients = normalized * self.target_std + self.target_mean
        return np.asarray(
            _expand_noise_basis(coefficients, action_horizon=ACTION_HORIZON),
            dtype=np.float32,
        )

    def _actor_reconstruction_mses(
        self,
        model_obs_items: list[Any],
        targets: list[np.ndarray],
        features: np.ndarray,
        states: np.ndarray,
    ) -> np.ndarray:
        if not model_obs_items:
            return np.zeros((0,), dtype=np.float32)
        noises = self._predict_raw_noise(features, states)
        mses = np.empty(len(model_obs_items), dtype=np.float32)
        for batch_start in range(0, len(model_obs_items), self.inversion_batch_size):
            batch_obs = jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0),
                *model_obs_items[batch_start:batch_start + self.inversion_batch_size],
            )
            batch_end = min(
                batch_start + self.inversion_batch_size, len(model_obs_items)
            )
            recon = self.inverter._denoise(
                batch_obs, jnp.asarray(noises[batch_start:batch_end])
            )
            target_batch = jnp.asarray(np.stack(targets[batch_start:batch_end], axis=0))
            mses[batch_start:batch_end] = action_reconstruction_mse(recon, target_batch)
        return mses

    def _batch(self, features, states, targets, indices):
        return freeze({
            "observations": {
                "pixels": jnp.asarray(features[indices, :, None]),
                "state": jnp.asarray(states[indices, :, None]),
            },
            "actions": jnp.asarray(targets[indices]),
        })

    def _demo_root_for_cache(self) -> Path | None:
        if self._online_context is None:
            return None
        return Path(self._online_context.get("demonstration_dir", self.demo_buffer_dir))

    def _load_expert_windows(self, episode_dir: Path):
        rows = []
        with (episode_dir / "steps.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
        by_sequence = {int(row["record_sequence"]): row for row in rows}
        experts = [row for row in rows if row.get("kind") == "expert"]
        boundaries = {
            int(row.get("intervention_segment_id", 0)): row
            for row in rows if row.get("kind") == "boundary"
        }
        groups: list[list[dict[str, Any]]] = []
        for row in experts:
            segment = int(row.get("intervention_segment_id", 0))
            if (
                not groups
                or segment != int(groups[-1][-1].get("intervention_segment_id", 0))
                or int(row["step_id"]) != int(groups[-1][-1]["step_id"]) + 1
            ):
                groups.append([])
            groups[-1].append(row)

        def following_policy(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
            segment_id = int(group[0].get("intervention_segment_id", 0))
            last_index = max(
                index for index, row in enumerate(rows)
                if row.get("kind") == "expert"
                and int(row.get("intervention_segment_id", 0)) == segment_id
            )
            following: list[dict[str, Any]] = []
            for row in rows[last_index + 1:]:
                kind = row.get("kind")
                if kind == "expert":
                    break
                if kind == "policy":
                    following.append(row)
            return following

        for group in groups:
            previous_state = None
            first_sequence = int(group[0]["record_sequence"])
            first_index = next(
                index for index, row in enumerate(rows)
                if int(row["record_sequence"]) == first_sequence
            )
            if first_index > 0:
                previous_state = rows[first_index - 1].get("state")
            group = trim_boundary_holds(group, previous_state=previous_state)
            if not group:
                continue
            segment = int(group[0].get("intervention_segment_id", 0))
            boundary = boundaries.get(segment)
            anchor = None
            if boundary is not None:
                anchor_sequence = boundary.get("policy_anchor_record_sequence")
                policy_anchor = (
                    by_sequence.get(int(anchor_sequence))
                    if anchor_sequence is not None else None
                )
                if (
                    policy_anchor is not None
                    and policy_anchor.get("kind") == "policy"
                    and 0.0 <= float(boundary["timestamp_s"]) - float(
                        policy_anchor["timestamp_s"]
                    ) <= 0.5
                    and policy_anchor.get("request_generation")
                    == boundary.get("request_generation")
                ):
                    anchor = policy_anchor
                else:
                    anchor = boundary
            policy_rows = following_policy(group)
            stop = len(group) if policy_rows else max(len(group) - 1, 0)
            stride = int(
                getattr(self, "window_stride", 0)
                or get_campaign_config().window_stride
            )
            for start in range(0, stop, stride):
                active_anchor = anchor if start == 0 and anchor is not None else group[start]
                future_start = start if active_anchor is anchor else start + 1
                future = group[future_start : future_start + ACTION_HORIZON]
                actions, pad_info = pad_actions_with_policy(
                    future, policy_rows
                )
                if not len(actions):
                    continue
                if int(pad_info.get("valid_length") or 0) < MIN_INTERVENTION_VALID_LENGTH:
                    continue
                observation = self._observation_from_row(episode_dir, active_anchor)
                yield observation, actions, {
                    "episode_id": episode_dir.name,
                    "start_step_id": int(active_anchor["step_id"]),
                    "intervention_segment_id": segment,
                    "anchor_kind": str(active_anchor.get("kind")),
                    "anchor_record_sequence": int(active_anchor["record_sequence"]),
                    **pad_info,
                }

    def _observation_from_row(
        self, episode_dir: Path, row: dict[str, Any]
    ) -> dict[str, Any]:
        images = {}
        for key, relative in row["images"].items():
            bgr = cv2.imread(str(episode_dir / relative), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(episode_dir / relative)
            images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return build_openpi_observation(
            images, row["state"], row.get("prompt") or self.default_prompt
        )

    def _load_autonomous_windows(self, episode_dir: Path):
        steps_path = Path(episode_dir) / "steps.jsonl"
        if not steps_path.is_file():
            return
        rows = []
        with steps_path.open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
        for row in policy_rows_without_later_intervention(rows):
            predicted = predicted_actions_from_row(row)
            if predicted is None or len(predicted) < ACTION_HORIZON:
                continue
            yield self._observation_from_row(episode_dir, row), predicted[:ACTION_HORIZON], {
                "episode_id": Path(episode_dir).name,
                "start_step_id": int(row["step_id"]),
                "anchor_kind": "policy",
                "anchor_record_sequence": int(row["record_sequence"]),
                "valid_length": int(ACTION_HORIZON),
                "policy_padded_length": 0,
                "padded_length": 0,
                "pad_source": "predicted_actions",
            }

    def _save_version(self, version: int, metadata: dict[str, Any]) -> None:
        versions = self.output_root / "steering_checkpoints"
        versions.mkdir(parents=True, exist_ok=True)
        final = versions / f"version_{version:06d}"
        tmp = versions / f".version_{version:06d}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        normalization = {
            "method": "per_dimension_zscore",
            "shape": [self.noise_basis_k, INTERNAL_ACTION_DIM],
            "deployment_clip_percentiles": [0.1, 99.9],
            "frozen": self._normalization_frozen,
            "mean": self.target_mean.tolist(),
            "std": self.target_std.tolist(),
            "clip_low": self.target_clip_low.tolist(),
            "clip_high": self.target_clip_high.tolist(),
        }
        self._atomic_json(tmp / "target_normalization.json", normalization)
        self._atomic_json(tmp / "steering_metadata.json", metadata)
        self.steering.save_checkpoint(str(tmp), step=version, keep_every_n_steps=1)
        os.replace(tmp, final)

    def _activate_version(self, version: int) -> None:
        versions = self.output_root / "steering_checkpoints"
        active_tmp = versions / ".active.tmp"
        active_tmp.write_text(str(version), encoding="utf-8")
        os.replace(active_tmp, versions / "ACTIVE")

    def _normalization_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.target_mean, self.target_std,
            self.target_clip_low, self.target_clip_high,
        ):
            if value is None:
                raise RuntimeError("normalization is incomplete")
            digest.update(np.asarray(value, dtype=np.float32).tobytes())
        return digest.hexdigest()

    def _restore_active_version(self) -> None:
        versions = self.output_root / "steering_checkpoints"
        active = versions / "ACTIVE"
        if not active.exists():
            return
        version = int(active.read_text(encoding="utf-8").strip())
        self.load_version(version)

    def load_version(self, version: int) -> int:
        """Load and pin one existing steering version without changing ACTIVE."""
        version = int(version)
        if version <= 0:
            raise ValueError("steering version must be a positive integer")
        versions = self.output_root / "steering_checkpoints"
        version_dir = versions / f"version_{version:06d}"
        normalization_path = version_dir / "target_normalization.json"
        metadata_path = version_dir / "steering_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(
                f"active steering version {version} has unknown base identity; "
                "legacy checkpoints are pipeline-only"
            )
        version_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if version_metadata.get("campaign_id") != self.campaign_id:
            raise RuntimeError("active steering belongs to a different lineage")
        if version_metadata.get("base_model_id") != self.base_model_id:
            raise RuntimeError(
                f"active steering version {version} base model does not match runtime"
            )
        if not normalization_path.is_file():
            raise RuntimeError(
                f"active steering version {version} is missing target normalization"
            )
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        expected_shape = (self.noise_basis_k, INTERNAL_ACTION_DIM)
        self.target_mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.target_std = np.asarray(normalization["std"], dtype=np.float32)
        self.target_clip_low = np.asarray(
            normalization["clip_low"], dtype=np.float32
        )
        self.target_clip_high = np.asarray(
            normalization["clip_high"], dtype=np.float32
        )
        for name, value in (
            ("mean", self.target_mean),
            ("std", self.target_std),
            ("clip_low", self.target_clip_low),
            ("clip_high", self.target_clip_high),
        ):
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"invalid target normalization {name}: shape={value.shape}"
                )
        if np.any(self.target_std <= 0) or np.any(
            self.target_clip_low > self.target_clip_high
        ):
            raise RuntimeError("invalid target normalization ranges")
        self._normalization_frozen = bool(normalization.get("frozen", True))
        self.steering = SteeringPolicy(
            seed=42,
            observations={
                "pixels": np.zeros((1, 2048, 1), dtype=np.float32),
                "state": np.zeros((1, 20, 1), dtype=np.float32),
            },
            actions=np.zeros((1, self.noise_basis_k, INTERNAL_ACTION_DIM), dtype=np.float32),
            lr=self.steering_lr,
            hidden_dims=(256, 256, 256),
            latent_dim=256,
            encoder_type="vlm_pi0",
            color_jitter=False,
            action_magnitude=3.0,
            num_cameras=3,
            output_bound="tanh",
        )
        self.steering.restore_checkpoint(str(version_dir))
        self.policy_version = version
        self.steering_eligible = True
        return version

    def mark_closed_loop_eligible(
        self, version: int, *, metrics: dict[str, Any], reload_max_abs: float
    ) -> Path:
        """Mark the active checkpoint usable. Online FlowDAgger does not gate this."""
        if version != self.policy_version:
            raise RuntimeError("can only attest the active steering version")
        eligibility = {
            "closed_loop_eligible": True,
            "policy_version": version,
            "base_model_id": self.base_model_id,
            "campaign_id": self.campaign_id,
        }
        path = (
            self.output_root / "steering_checkpoints" / "eligible"
            / f"version_{version:06d}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(path, eligibility)
        self.steering_eligible = True
        return path

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
