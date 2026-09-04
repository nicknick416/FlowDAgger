"""Replay buffer: intervention, autonomous, and prior demonstration sources."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def classify(metadata: dict[str, Any]) -> tuple[str, str, str]:
    outcome = metadata.get("task_outcome")
    mode = metadata.get("completion_mode")
    if outcome in ("success", "failure", "abort") and mode in (
        "autonomous", "assisted"
    ):
        return str(outcome), str(mode), "protocol_v3"
    label = str(metadata.get("label", ""))
    interventions = int(
        metadata.get("episode_metrics", {}).get("intervention_count", 0)
    )
    if label in ("assisted_success", "autonomous_success", "success"):
        return (
            "success",
            "assisted" if interventions > 0 or label == "assisted_success" else "autonomous",
            "inferred_legacy",
        )
    if label in ("failure", "abort"):
        return label, "assisted" if interventions > 0 else "autonomous", "inferred_legacy"
    raise ValueError(f"cannot classify episode metadata: label={label!r}")


def collect_online_sources(output_root: Path, current: Path) -> dict[str, list[Path]]:
    """Split this campaign's successes into intervention and autonomous pools."""
    current = Path(current).resolve()
    intervention_history: list[Path] = []
    autonomous_history: list[Path] = []
    episodes_root = Path(output_root) / "episodes"
    if episodes_root.is_dir():
        for path in sorted(episodes_root.glob("episode_*")):
            if path.resolve() == current:
                continue
            metadata_path = path / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            outcome, mode, _ = classify(metadata)
            if outcome != "success":
                continue
            autonomous_history.append(path.resolve())
            if mode == "assisted":
                intervention_history.append(path.resolve())
    return {
        "current": [current],
        "intervention_history": intervention_history,
        "autonomous": [current, *autonomous_history],
        "history": intervention_history,
    }
