"""Disk cache for frozen-base flow-matching inversions.

The inverter is deterministic given (frozen pi0.5, obs, 50-step actions,
inverter hyperparams). Raw coefficients and VLM features can be reused
across online BC rounds. Z-score targets are NOT cached: they are refit
until normalization freezes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


CACHE_VERSION = 1
CACHE_META_NAME = "inversion_cache.json"
CACHE_ARRAY_NAME = "inversion_cache.npz"


@dataclass
class CachedInversion:
    accepted: bool
    mse: float
    e_repr: float
    feature: np.ndarray | None = None
    state: np.ndarray | None = None
    coefficients: np.ndarray | None = None


def window_cache_key(info: Mapping[str, Any]) -> str:
    return "|".join([
        str(info.get("source", "")),
        str(info.get("start_step_id", "")),
        str(info.get("anchor_kind", "")),
        str(info.get("anchor_record_sequence", "")),
        str(info.get("intervention_segment_id", "")),
    ])


def inversion_cache_dir(
    episode_dir: str | Path,
    *,
    demo_root: str | Path | None = None,
) -> Path:
    """Campaign episodes cache in-place; demo symlinks cache under demo_root."""
    episode_dir = Path(episode_dir)
    if demo_root is not None:
        demo_root = Path(demo_root).resolve()
        if episode_dir.parent.resolve() == demo_root:
            return demo_root / "inversion_cache" / episode_dir.name
        linked = demo_root / episode_dir.name
        if linked.exists() and linked.resolve() == episode_dir.resolve():
            return demo_root / "inversion_cache" / episode_dir.name
    return episode_dir / "inversion_cache"


def inversion_cache_compatible(
    meta: Mapping[str, Any],
    *,
    base_model_id: str,
    noise_basis_k: int,
    inversion_mse_threshold: float,
) -> bool:
    return (
        int(meta.get("cache_version", -1)) == CACHE_VERSION
        and str(meta.get("base_model_id", "")) == str(base_model_id)
        and int(meta.get("noise_basis_k", -1)) == int(noise_basis_k)
        and abs(
            float(meta.get("inversion_mse_threshold", -1.0))
            - float(inversion_mse_threshold)
        ) < 1e-12
    )


def load_episode_cache(
    cache_dir: str | Path,
    *,
    base_model_id: str,
    noise_basis_k: int,
    inversion_mse_threshold: float,
) -> dict[str, CachedInversion]:
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / CACHE_META_NAME
    array_path = cache_dir / CACHE_ARRAY_NAME
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not inversion_cache_compatible(
        meta,
        base_model_id=base_model_id,
        noise_basis_k=noise_basis_k,
        inversion_mse_threshold=inversion_mse_threshold,
    ):
        return {}
    accepted_by_key: dict[str, dict[str, np.ndarray]] = {}
    accepted_keys = [
        str(key) for key in meta.get("accepted_keys", [])
    ]
    if accepted_keys and array_path.is_file():
        with np.load(array_path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            states = np.asarray(payload["states"], dtype=np.float32)
            coefficients = np.asarray(payload["coefficients"], dtype=np.float32)
        if (
            len(features) != len(accepted_keys)
            or len(states) != len(accepted_keys)
            or len(coefficients) != len(accepted_keys)
        ):
            return {}
        for index, key in enumerate(accepted_keys):
            accepted_by_key[key] = {
                "feature": features[index],
                "state": states[index],
                "coefficients": coefficients[index],
            }
    records: dict[str, CachedInversion] = {}
    for key, row in dict(meta.get("windows", {})).items():
        accepted = bool(row.get("accepted"))
        arrays = accepted_by_key.get(str(key))
        if accepted and arrays is None:
            continue
        records[str(key)] = CachedInversion(
            accepted=accepted,
            mse=float(row.get("mse", np.nan)),
            e_repr=float(row.get("e_repr", row.get("mse", np.nan))),
            feature=None if arrays is None else arrays["feature"],
            state=None if arrays is None else arrays["state"],
            coefficients=None if arrays is None else arrays["coefficients"],
        )
    return records


def save_episode_cache(
    cache_dir: str | Path,
    records: Mapping[str, CachedInversion],
    *,
    base_model_id: str,
    noise_basis_k: int,
    inversion_mse_threshold: float,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    accepted_keys = [
        key for key, record in records.items() if record.accepted
    ]
    meta = {
        "cache_version": CACHE_VERSION,
        "base_model_id": str(base_model_id),
        "noise_basis_k": int(noise_basis_k),
        "inversion_mse_threshold": float(inversion_mse_threshold),
        "accepted_keys": accepted_keys,
        "windows": {
            key: {
                "accepted": bool(record.accepted),
                "mse": float(record.mse),
                "e_repr": float(record.e_repr),
            }
            for key, record in records.items()
        },
    }
    tmp_meta = cache_dir / f".{CACHE_META_NAME}.tmp"
    tmp_arrays = cache_dir / f".{CACHE_ARRAY_NAME}.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if accepted_keys:
        with tmp_arrays.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.stack(
                    [
                        np.asarray(records[key].feature, dtype=np.float32)
                        for key in accepted_keys
                    ]
                ),
                states=np.stack(
                    [
                        np.asarray(records[key].state, dtype=np.float32)
                        for key in accepted_keys
                    ]
                ),
                coefficients=np.stack(
                    [
                        np.asarray(records[key].coefficients, dtype=np.float32)
                        for key in accepted_keys
                    ]
                ),
            )
        os.replace(tmp_arrays, cache_dir / CACHE_ARRAY_NAME)
    elif (cache_dir / CACHE_ARRAY_NAME).exists():
        (cache_dir / CACHE_ARRAY_NAME).unlink()
        if tmp_arrays.exists():
            tmp_arrays.unlink()
    os.replace(tmp_meta, cache_dir / CACHE_META_NAME)


def split_windows_by_cache(
    windows: Iterable[tuple[Any, Any, dict[str, Any]]],
    load_records: Callable[[str], Mapping[str, CachedInversion]],
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]],
    list[tuple[Any, Any, dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Return (cached accepted samples, cache misses, cache-hit reports)."""
    cached_samples: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]
    ] = []
    missed: list[tuple[Any, Any, dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    memo: dict[str, Mapping[str, CachedInversion]] = {}
    for observation, actions, window_info in windows:
        episode_path = str(window_info["episode_path"])
        if episode_path not in memo:
            memo[episode_path] = load_records(episode_path)
        record = memo[episode_path].get(window_cache_key(window_info))
        if record is None:
            missed.append((observation, actions, window_info))
            continue
        report = {
            **window_info,
            "normalized_action_mse": float(record.mse),
            "e_full": float(record.mse),
            "e_repr": float(record.e_repr),
            "e_actor_before": None,
            "e_actor_after": None,
            "accepted": bool(record.accepted),
            "from_cache": True,
        }
        if record.accepted:
            info = dict(window_info)
            info["from_cache"] = True
            cached_samples.append((
                np.asarray(record.feature, dtype=np.float32),
                np.asarray(record.state, dtype=np.float32),
                np.asarray(record.coefficients, dtype=np.float32),
                float(record.mse),
                info,
            ))
            report["sample_index"] = len(cached_samples) - 1
        reports.append(report)
    return cached_samples, missed, reports
