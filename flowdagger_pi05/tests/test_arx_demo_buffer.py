import json
from pathlib import Path

from arx_demo_buffer import (
    copy_demonstrations,
    expert_window_starts,
    select_shortest_demonstrations,
)


def _raw_episode(root: Path, name: str, *, duration: float, frames: int) -> Path:
    path = root / "20260829" / "op" / name
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({
        "episode_id": name,
        "duration_seconds": duration,
        "total_frames": frames,
        "quality": {"labels": ["完全正常"]},
    }))
    return path


def test_expert_window_starts_leave_full_horizon():
    assert expert_window_starts(51) == [0]
    assert expert_window_starts(71) == [0, 10, 20]
    assert expert_window_starts(50) == []


def test_select_shortest_demonstrations_orders_by_duration(tmp_path):
    raw = tmp_path / "raw"
    _raw_episode(raw, "episode_slow", duration=40.0, frames=200)
    _raw_episode(raw, "episode_fast", duration=18.2, frames=546)
    _raw_episode(raw, "episode_mid", duration=19.5, frames=585)
    _raw_episode(raw, "episode_tiny", duration=1.0, frames=20)
    selected = select_shortest_demonstrations(raw, count=2)
    assert [row["episode_id"] for row in selected] == [
        "episode_fast", "episode_mid",
    ]
    output = copy_demonstrations(selected, tmp_path / "buffer")
    payload = json.loads((output / "manifest.json").read_text())
    assert payload["count"] == 2
    assert (output / "episode_fast").is_symlink()
