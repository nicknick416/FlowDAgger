"""Evaluate staged ARX FlowDAgger episodes against the deployment gates."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from arx_campaign import add_config_file_arg, preload_campaign_config


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    episodes = []
    for directory in sorted((root / "episodes").glob("*")):
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("runtime_mode") == "protocol_only":
            continue
        # Backward-compatible exclusion for protocol episodes recorded before
        # runtime_mode was added to metadata.
        steps_path = directory / "steps.jsonl"
        if steps_path.exists() and any(
            json.loads(line).get("executed_policy") == "protocol_only"
            for line in steps_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ):
            continue
        metadata["episode_dir"] = str(directory)
        episodes.append(metadata)
    return episodes


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _control(episode: dict[str, Any]) -> dict[str, Any]:
    return episode.get("episode_metrics", {}).get("control", {})


def _expert_capture_metrics(episode: dict[str, Any]) -> dict[str, float | int]:
    directory = Path(episode["episode_dir"])
    steps_path = directory / "steps.jsonl"
    expert = []
    if steps_path.exists():
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "expert":
                expert.append(row)
    intervals = [
        float(right["timestamp_s"]) - float(left["timestamp_s"])
        for left, right in zip(expert, expert[1:])
        if (
            int(right["step_id"]) == int(left["step_id"]) + 1
            and float(right["timestamp_s"]) > float(left["timestamp_s"])
        )
    ]
    fps = (len(intervals) / sum(intervals)) if intervals and sum(intervals) > 0 else 0.0
    complete_images = sum(len(row.get("images", {})) == 3 for row in expert)
    return {"expert_steps": len(expert), "complete_images": complete_images, "fps": fps}


def _success_rate(episodes: list[dict[str, Any]]) -> float:
    success_labels = {"assisted_success", "autonomous_success"}
    return sum(item.get("label") in success_labels for item in episodes) / len(episodes)


def main() -> None:
    cfg = preload_campaign_config()
    parser = argparse.ArgumentParser()
    add_config_file_arg(parser)
    parser.add_argument(
        "--output-root",
        default=cfg.output_root,
    )
    parser.add_argument(
        "--gate",
        choices=("demonstration", "shadow", "closed_loop", "all"),
        default="all",
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    episodes = _load_episodes(root)
    by_stage = {
        stage: [item for item in episodes if item.get("run_stage") == stage]
        for stage in ("demonstration", "baseline", "shadow", "closed_loop")
    }
    failures: list[str] = []

    if args.gate in ("demonstration", "all"):
        successful = [
            item
            for item in by_stage["demonstration"]
            if item.get("label") == "assisted_success"
        ]
        if len(successful) < 5:
            failures.append(
                f"demonstration assisted-success episodes {len(successful)} < 5"
            )
        if any(item.get("steering_policy_version", 0) != 0 for item in successful):
            failures.append("demonstration stage unexpectedly used a steering checkpoint")
        for item in successful:
            capture = _expert_capture_metrics(item)
            if capture["expert_steps"] < 2:
                failures.append(f"{item['episode_id']} has fewer than 2 expert samples")
            if capture["complete_images"] != capture["expert_steps"]:
                failures.append(f"{item['episode_id']} is missing synchronized camera frames")

    if args.gate in ("shadow", "all"):
        shadow = by_stage["shadow"][-5:]
        if len(shadow) < 5:
            failures.append(f"shadow episodes {len(shadow)} < 5")
        for item in shadow:
            metrics = item.get("episode_metrics", {})
            if item.get("shadow_mode") is not True:
                failures.append(f"{item['episode_id']} did not run with shadow_mode=true")
            if metrics.get("shadow_observations", 0) <= 0:
                failures.append(f"{item['episode_id']} has no shadow comparisons")

    closed_summary = None
    if args.gate in ("closed_loop", "all"):
        baseline = by_stage["baseline"][-10:]
        closed = by_stage["closed_loop"][-10:]
        if len(baseline) < 10:
            failures.append(f"baseline episodes {len(baseline)} < 10")
        if len(closed) < 10:
            failures.append(f"closed-loop episodes {len(closed)} < 10")
        if len(baseline) == 10 and len(closed) == 10:
            baseline_rate = _success_rate(baseline)
            closed_rate = _success_rate(closed)
            first_intervention = sum(
                float(_control(item).get("intervention_step_fraction", 0.0))
                for item in closed[:5]
            ) / 5
            last_intervention = sum(
                float(_control(item).get("intervention_step_fraction", 0.0))
                for item in closed[5:]
            ) / 5
            closed_summary = {
                "baseline_success_rate": baseline_rate,
                "closed_loop_success_rate": closed_rate,
                "first_5_intervention_fraction": first_intervention,
                "last_5_intervention_fraction": last_intervention,
            }
            if closed_rate < baseline_rate:
                failures.append(
                    f"closed-loop success rate {closed_rate:.3f} < baseline {baseline_rate:.3f}"
                )
            if last_intervention > first_intervention:
                failures.append(
                    "last-5 intervention fraction exceeds first-5: "
                    f"{last_intervention:.3f} > {first_intervention:.3f}"
                )
            for item in closed:
                control = _control(item)
                if control.get("safety_rejections", 0) or control.get(
                    "action_rejections", 0
                ):
                    failures.append(f"{item['episode_id']} contains a safety anomaly")
                if item.get("shadow_mode") is not False:
                    failures.append(f"{item['episode_id']} was not a closed-loop execution")

    if not _finite_tree(episodes):
        failures.append("episode metadata contains NaN/Inf")
    report = {
        "gate": args.gate,
        "episode_counts": {key: len(value) for key, value in by_stage.items()},
        "closed_loop": closed_summary,
        "failures": failures,
        "passed": not failures,
    }
    report_dir = root / "acceptance"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.gate}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
