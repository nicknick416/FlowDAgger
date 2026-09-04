"""Select the shortest raw expert episodes and load them as demonstration windows."""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from arx_adapter import (
    ACTION_DIM,
    ACTION_HORIZON,
    CAMERA_KEYS,
    DEMO_BUFFER_COUNT,
    DEMO_BUFFER_DIR,
    RAW_DATA_ROOT,
    WINDOW_STRIDE,
    build_openpi_observation,
)


RAW_CAMERA_TO_OBS = {
    "observation.image.third_view": "observation/image",
    "observation.image.left_wrist_view": "observation/left_wrist_image",
    "observation.image.right_wrist_view": "observation/right_wrist_image",
}
STATE_COLUMNS = [
    "left_x", "left_y", "left_z",
    "left_r1", "left_r2", "left_r3", "left_r4", "left_r5", "left_r6",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_r1", "right_r2", "right_r3", "right_r4", "right_r5", "right_r6",
    "right_gripper",
]


def expert_window_starts(
    n_frames: int,
    *,
    horizon: int = ACTION_HORIZON,
    stride: int = WINDOW_STRIDE,
) -> list[int]:
    last_start = int(n_frames) - int(horizon) - 1
    if last_start < 0:
        return []
    return list(range(0, last_start + 1, int(stride)))


def iter_raw_episode_dirs(raw_root: str | Path = RAW_DATA_ROOT) -> Iterator[Path]:
    root = Path(raw_root)
    for date_dir in sorted(root.glob("20*")):
        if not date_dir.is_dir():
            continue
        for episode_dir in sorted(date_dir.rglob("episode_*")):
            if episode_dir.is_dir() and (episode_dir / "metadata.json").is_file():
                yield episode_dir


def select_shortest_demonstrations(
    raw_root: str | Path = RAW_DATA_ROOT,
    *,
    count: int = DEMO_BUFFER_COUNT,
    min_frames: int = ACTION_HORIZON + 1,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for episode_dir in iter_raw_episode_dirs(raw_root):
        metadata = json.loads(
            (episode_dir / "metadata.json").read_text(encoding="utf-8")
        )
        labels = [
            str(item) for item in metadata.get("quality", {}).get("labels", [])
        ]
        total_frames = int(metadata.get("total_frames") or 0)
        if "完全正常" not in labels or total_frames < int(min_frames):
            continue
        selected.append({
            "episode_id": str(metadata.get("episode_id") or episode_dir.name),
            "path": str(episode_dir.resolve()),
            "duration_seconds": float(metadata.get("duration_seconds") or 0.0),
            "total_frames": total_frames,
        })
    selected.sort(
        key=lambda row: (
            float(row["duration_seconds"]),
            int(row["total_frames"]),
            str(row["episode_id"]),
        )
    )
    return selected[: int(count)]


def copy_demonstrations(
    rows: list[dict[str, Any]],
    output_dir: str | Path = DEMO_BUFFER_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in rows:
        source = Path(row["path"])
        destination = output / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source, destination)
        copied.append({**row, "path": str(destination)})
    manifest = {
        "selection": "shortest_duration",
        "count": len(copied),
        "episodes": copied,
    }
    path = output / "manifest.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return output


def iter_demo_episode_dirs(output_dir: str | Path = DEMO_BUFFER_DIR) -> Iterator[Path]:
    root = Path(output_dir)
    for path in sorted(root.glob("episode_*")):
        if path.is_dir() and (path / "metadata.json").is_file():
            yield path


def load_eef_states(episode_dir: str | Path) -> np.ndarray:
    path = Path(episode_dir) / "observation.state.eef_pose" / "data.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing = [column for column in STATE_COLUMNS if column not in (rows[0] if rows else {})]
    if not rows or missing:
        raise ValueError(f"invalid eef pose table: {path}")
    values = np.asarray(
        [[float(row[column]) for column in STATE_COLUMNS] for row in rows],
        dtype=np.float32,
    )
    if values.shape[1] != ACTION_DIM or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid eef states in {path}")
    return values


def iter_raw_expert_windows(
    episode_dir: str | Path,
    *,
    prompt: str,
    horizon: int = ACTION_HORIZON,
    stride: int = 50,
) -> Iterator[tuple[dict[str, Any], np.ndarray, dict[str, Any]]]:
    import cv2

    episode_dir = Path(episode_dir)
    states = load_eef_states(episode_dir)
    captures = {}
    for raw_key, obs_key in RAW_CAMERA_TO_OBS.items():
        capture = cv2.VideoCapture(str(episode_dir / raw_key / "video.mp4"))
        if not capture.isOpened():
            raise FileNotFoundError(episode_dir / raw_key / "video.mp4")
        captures[obs_key] = capture
    try:
        image_ring: deque[dict[str, np.ndarray]] = deque(maxlen=horizon + 1)
        for frame_index in range(len(states)):
            images = {}
            ok = True
            for key in CAMERA_KEYS:
                read_ok, frame = captures[key].read()
                if not read_ok or frame is None:
                    ok = False
                    break
                images[key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ok:
                break
            image_ring.append(images)
            start = frame_index - horizon
            if start < 0 or start % stride != 0:
                continue
            actions = states[start + 1 : start + 1 + horizon]
            if len(actions) != horizon:
                continue
            yield build_openpi_observation(
                image_ring[0], states[start], prompt
            ), np.asarray(actions, dtype=np.float32), {
                "episode_id": episode_dir.name,
                "start_step_id": int(start),
                "anchor_kind": "expert",
                "valid_length": int(horizon),
                "policy_padded_length": 0,
                "padded_length": 0,
                "pad_source": "none",
                "source": "demonstration",
                "origin": "prior",
            }
    finally:
        for capture in captures.values():
            capture.release()
