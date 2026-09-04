"""Crash-safe on-disk episode recording for the ARX FlowDAgger service."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from arx_adapter import CAMERA_KEYS, validate_arx_state


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_NUMBERED_EPISODE = re.compile(
    r"^episode_\d{8}_(?P<sequence>\d{4,})_\d{6}$"
)


class EpisodeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.partial_root = self.root / ".partial"
        self.episodes_root = self.root / "episodes"
        self.partial_root.mkdir(parents=True, exist_ok=True)
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_id: str | None = None
        self._metadata: dict[str, Any] = {}
        self._record_sequence = 0
        self._last_expert_step_id: int | None = None
        self._intervention_segment_id = 0
        self._last_policy_record_sequence: int | None = None
        self._summary: dict[str, int | float] = {}

    @property
    def active_episode_id(self) -> str | None:
        return self._active_id

    @property
    def active_summary(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._summary)

    def start(self, episode_id: str, **metadata: Any) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(episode_id):
            raise ValueError("episode_id contains unsafe characters")
        with self._lock:
            if self._active_id is not None:
                raise RuntimeError(f"episode {self._active_id} is already active")
            partial = self.partial_root / episode_id
            if partial.exists() or (self.episodes_root / episode_id).exists():
                raise FileExistsError(f"episode already exists: {episode_id}")
            (partial / "frames").mkdir(parents=True)
            self._active_id = episode_id
            self._record_sequence = 0
            self._last_expert_step_id = None
            self._intervention_segment_id = 0
            self._last_policy_record_sequence = None
            self._summary = {
                "policy_observations": 0,
                "expert_observations": 0,
                "expert_transitions": 0,
                "intervention_count": 0,
                "shadow_observations": 0,
                "shadow_action_mse_sum": 0.0,
                "shadow_action_max_abs": 0.0,
            }
            self._metadata = {
                "episode_id": episode_id,
                "started_at_s": time.time(),
                "format_version": 3,
                **metadata,
            }
            self._write_json(partial / "metadata.json", self._metadata)
            return dict(self._metadata)

    def append_event(self, kind: str, **fields: Any) -> None:
        with self._lock:
            partial = self._require_active()
            if kind == "intervention_start":
                self._summary["intervention_count"] += 1
                self._intervention_segment_id += 1
                self._last_expert_step_id = None
                fields.setdefault("intervention_segment_id", self._intervention_segment_id)
                fields.setdefault(
                    "policy_anchor_record_sequence", self._last_policy_record_sequence
                )
            self._append_jsonl(partial / "events.jsonl", {
                "timestamp_s": time.time(),
                "kind": kind,
                **fields,
            })

    def append_observation(
        self,
        *,
        kind: str,
        step_id: int,
        images: Mapping[str, np.ndarray],
        state: Any,
        prompt: str,
        **fields: Any,
    ) -> int:
        with self._lock:
            partial = self._require_active()
            state20 = validate_arx_state(state)
            observation_key = f"{kind}_observations"
            if observation_key in self._summary:
                self._summary[observation_key] += 1
            if kind == "expert":
                current_step_id = int(step_id)
                if (
                    self._last_expert_step_id is not None
                    and current_step_id == self._last_expert_step_id + 1
                ):
                    self._summary["expert_transitions"] += 1
                self._last_expert_step_id = current_step_id
            if "shadow_action_mse" in fields:
                self._summary["shadow_observations"] += 1
                self._summary["shadow_action_mse_sum"] += float(
                    fields["shadow_action_mse"]
                )
                self._summary["shadow_action_max_abs"] = max(
                    float(self._summary["shadow_action_max_abs"]),
                    float(fields.get("shadow_action_max_abs", 0.0)),
                )
            record_sequence = self._record_sequence
            self._record_sequence += 1
            if kind == "policy":
                self._last_policy_record_sequence = record_sequence
            if kind in ("boundary", "expert"):
                fields.setdefault(
                    "intervention_segment_id", self._intervention_segment_id
                )
                fields.setdefault(
                    "policy_anchor_record_sequence", self._last_policy_record_sequence
                )
            frame_dir = partial / "frames" / (
                f"{record_sequence:010d}_{int(step_id):08d}_{kind}"
            )
            frame_dir.mkdir(parents=True, exist_ok=False)
            image_paths: dict[str, str] = {}
            image_shapes: dict[str, list[int]] = {}
            for key in CAMERA_KEYS:
                if key not in images:
                    raise ValueError(f"missing image key {key}")
                name = key.replace("observation/", "").replace("/", "_") + ".jpg"
                path = frame_dir / name
                image = np.asarray(images[key], dtype=np.uint8)
                # Server images are RGB CHW; OpenCV writes BGR HWC.
                if image.ndim == 3 and image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
                image_shapes[key] = [int(value) for value in image.shape]
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"failed to write {path}")
                image_paths[key] = str(path.relative_to(partial))
            timestamp_s = fields.pop("timestamp_s", None)
            self._append_jsonl(partial / "steps.jsonl", {
                "timestamp_s": float(timestamp_s) if timestamp_s is not None else time.time(),
                "kind": kind,
                "record_sequence": record_sequence,
                "step_id": int(step_id),
                "state": state20.tolist(),
                "prompt": str(prompt),
                "images": image_paths,
                "image_shapes": image_shapes,
                **fields,
            })
            return record_sequence

    def finish(self, task_outcome: str, **metadata: Any) -> Path:
        if task_outcome not in ("success", "failure", "abort"):
            raise ValueError(f"invalid task_outcome: {task_outcome}")
        with self._lock:
            partial = self._require_active()
            episode_id = self._active_id
            shadow_count = int(self._summary.get("shadow_observations", 0))
            episode_metrics = dict(self._summary)
            mse_sum = float(episode_metrics.pop("shadow_action_mse_sum", 0.0))
            episode_metrics["shadow_action_mse_mean"] = (
                mse_sum / shadow_count if shadow_count else None
            )
            supplied_metrics = metadata.pop("control_metrics", {})
            if supplied_metrics:
                control = dict(supplied_metrics)
                policy_steps = int(control.get("policy_steps", 0))
                expert_steps = int(control.get("expert_steps", 0))
                total_control_steps = policy_steps + expert_steps
                control["intervention_step_fraction"] = (
                    expert_steps / total_control_steps if total_control_steps else 0.0
                )
                control["eef_clamp_trigger_rate"] = (
                    float(control.get("eef_step_clamps", 0)) / policy_steps
                    if policy_steps
                    else 0.0
                )
                episode_metrics["control"] = control
            final_name = self._next_episode_dir_name(
                float(self._metadata["started_at_s"])
            )
            source_episode_id = str(episode_id)
            intervention_count = int(self._summary.get("intervention_count", 0))
            completion_mode = "assisted" if intervention_count > 0 else "autonomous"
            label = (
                f"{completion_mode}_success"
                if task_outcome == "success"
                else task_outcome
            )
            self._metadata.update({
                "source_episode_id": source_episode_id,
                "episode_id": final_name,
                "label": label,
                "task_outcome": task_outcome,
                "completion_mode": completion_mode,
                "classification_source": "protocol_v3_automatic",
                "finished_at_s": time.time(),
                "episode_metrics": episode_metrics,
                **metadata,
            })
            self._write_json(partial / "metadata.json", self._metadata)
            final = self.episodes_root / final_name
            os.replace(partial, final)
            self._active_id = None
            self._metadata = {}
            self._record_sequence = 0
            self._summary = {}
            return final

    def _next_episode_dir_name(self, started_at_s: float) -> str:
        """Allocate a stable folder-wide sequence for a completed episode."""
        episode_dirs = [
            path
            for path in self.episodes_root.iterdir()
            if path.is_dir() and path.name.startswith("episode_")
        ]
        max_numbered_sequence = 0
        for path in episode_dirs:
            match = _NUMBERED_EPISODE.fullmatch(path.name)
            if match:
                max_numbered_sequence = max(
                    max_numbered_sequence, int(match.group("sequence"))
                )
        # Count legacy episode_* folders too, so migration does not restart at 1.
        sequence = max(len(episode_dirs) + 1, max_numbered_sequence + 1)
        capture_time = datetime.fromtimestamp(started_at_s)
        name = (
            f"episode_{capture_time:%Y%m%d}_{sequence:04d}_{capture_time:%H%M%S}"
        )
        if (self.episodes_root / name).exists():
            raise FileExistsError(f"episode already exists: {name}")
        return name

    def abort_incomplete(self, reason: str) -> Path | None:
        with self._lock:
            if self._active_id is None:
                return None
            return self.finish("abort", abort_reason=reason)

    def _require_active(self) -> Path:
        if self._active_id is None:
            raise RuntimeError("no active episode")
        return self.partial_root / self._active_id

    @staticmethod
    def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
