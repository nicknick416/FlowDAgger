import json
from pathlib import Path

from arx_replay import classify, collect_online_sources


def _episode(root: Path, name: str, *, assisted: bool, success: bool = True) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({
        "label": (
            "assisted_success" if assisted and success
            else "autonomous_success" if success
            else "failure"
        ),
        "task_outcome": "success" if success else "failure",
        "completion_mode": "assisted" if assisted else "autonomous",
        "episode_metrics": {"intervention_count": 1 if assisted else 0},
    }))
    return path


def test_replay_uses_campaign_assisted_successes_without_manifest(tmp_path):
    campaign = tmp_path / "campaign"
    episodes = campaign / "episodes"
    history = [_episode(episodes, f"episode_{index}", assisted=True) for index in range(3)]
    autonomous = _episode(episodes, "episode_auto", assisted=False, success=True)
    _episode(episodes, "episode_fail", assisted=False, success=False)
    current = _episode(episodes, "episode_current", assisted=True)
    sources = collect_online_sources(campaign, current)
    assert sources["current"] == [current.resolve()]
    assert sources["intervention_history"] == [path.resolve() for path in history]
    assert sources["history"] == sources["intervention_history"]
    assert sources["autonomous"] == [
        current.resolve(),
        *[path.resolve() for path in history],
        autonomous.resolve(),
    ]


def test_classify_assisted_success():
    outcome, mode, source = classify({
        "task_outcome": "success",
        "completion_mode": "assisted",
    })
    assert (outcome, mode, source) == ("success", "assisted", "protocol_v3")
