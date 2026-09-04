import numpy as np
from pathlib import Path

from arx_adapter import (
    resolve_base_model_identity,
)
from arx_campaign import get_campaign_config
from arx_inversion_cache import (
    CachedInversion,
    inversion_cache_dir,
    load_episode_cache,
    save_episode_cache,
    split_windows_by_cache,
    window_cache_key,
)


def _record(*, accepted: bool = True, mse: float = 1e-4) -> CachedInversion:
    feature = np.arange(4, dtype=np.float32)
    state = np.ones(20, dtype=np.float32)
    coefficients = np.full((50, 32), 0.2, dtype=np.float32)
    return CachedInversion(
        accepted=accepted,
        mse=mse,
        e_repr=mse,
        feature=feature if accepted else None,
        state=state if accepted else None,
        coefficients=coefficients if accepted else None,
    )


def test_window_cache_key_is_stable():
    info = {
        "source": "intervention",
        "start_step_id": 12,
        "anchor_kind": "policy",
        "anchor_record_sequence": 8,
        "intervention_segment_id": 1,
    }
    assert window_cache_key(info) == "intervention|12|policy|8|1"
    assert window_cache_key(info) == window_cache_key(dict(info))


def test_demo_cache_stays_under_demo_root(tmp_path):
    raw = tmp_path / "raw" / "episode_fast"
    raw.mkdir(parents=True)
    demo_root = tmp_path / "buffer"
    demo_root.mkdir()
    linked = demo_root / "episode_fast"
    linked.symlink_to(raw)
    expected = demo_root / "inversion_cache" / "episode_fast"
    assert inversion_cache_dir(linked, demo_root=demo_root) == expected
    assert inversion_cache_dir(raw.resolve(), demo_root=demo_root) == expected
    assert expected.resolve() != raw.resolve()


def test_campaign_cache_lives_in_episode_dir(tmp_path):
    episode = tmp_path / "episodes" / "ep1"
    episode.mkdir(parents=True)
    assert inversion_cache_dir(episode) == episode / "inversion_cache"


def test_inversion_cache_roundtrip_and_base_mismatch(tmp_path):
    cache_dir = tmp_path / "ep" / "inversion_cache"
    accepted = _record()
    rejected = _record(accepted=False, mse=9.0)
    save_episode_cache(
        cache_dir,
        {
            "intervention|0|expert||1": accepted,
            "intervention|10|expert||1": rejected,
        },
        base_model_id="connect_elevator_pins_arx_0901:20000:abcd",
        noise_basis_k=50,
        inversion_mse_threshold=1e-3,
    )
    loaded = load_episode_cache(
        cache_dir,
        base_model_id="connect_elevator_pins_arx_0901:20000:abcd",
        noise_basis_k=50,
        inversion_mse_threshold=1e-3,
    )
    assert loaded["intervention|0|expert||1"].accepted
    np.testing.assert_allclose(
        loaded["intervention|0|expert||1"].coefficients, accepted.coefficients
    )
    assert loaded["intervention|10|expert||1"].accepted is False
    stale = load_episode_cache(
        cache_dir,
        base_model_id="old-base",
        noise_basis_k=50,
        inversion_mse_threshold=1e-3,
    )
    assert stale == {}


def test_split_windows_skips_cached_and_keeps_misses():
    cached = _record()
    windows = [
        ({}, np.zeros((50, 20), np.float32), {
            "episode_path": "/ep/a",
            "episode_id": "a",
            "source": "intervention",
            "start_step_id": 0,
            "anchor_kind": "expert",
            "anchor_record_sequence": "",
            "intervention_segment_id": 1,
        }),
        ({}, np.ones((50, 20), np.float32), {
            "episode_path": "/ep/a",
            "episode_id": "a",
            "source": "autonomous",
            "start_step_id": 10,
            "anchor_kind": "policy",
            "anchor_record_sequence": 4,
            "intervention_segment_id": "",
        }),
    ]
    records = {
        "/ep/a": {window_cache_key(windows[0][2]): cached},
    }
    samples, missed, reports = split_windows_by_cache(
        windows, lambda path: records.get(path, {})
    )
    assert len(samples) == 1
    assert samples[0][4]["from_cache"] is True
    assert len(missed) == 1
    assert missed[0][2]["start_step_id"] == 10
    assert reports[0]["from_cache"] is True
    assert reports[0]["accepted"] is True


def test_base_identity_matches_campaign_config():
    cfg = get_campaign_config()
    checkpoint = Path(cfg.base_checkpoint)
    assert checkpoint.name == str(cfg.base_checkpoint_step)
    assert checkpoint.parent.name == cfg.base_checkpoint_name
    assert cfg.base_asset_id == cfg.base_checkpoint_name
    assert Path(cfg.output_root).name == cfg.campaign_id
    if checkpoint.is_dir():
        identity = resolve_base_model_identity(cfg.base_checkpoint)
        assert identity["base_model_id"].startswith(
            f"{cfg.base_checkpoint_name}:{cfg.base_checkpoint_step}:"
        )
        assert identity["asset_id"] == cfg.base_asset_id
