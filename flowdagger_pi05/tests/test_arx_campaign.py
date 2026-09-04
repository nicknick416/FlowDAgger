from pathlib import Path

import pytest
import yaml

from arx_campaign import (
    campaign_config_from_mapping,
    get_campaign_config,
    load_campaign_config,
    set_campaign_config,
)


def _payload(**overrides):
    data = {
        "campaign_id": "arx_test_v1",
        "runs_root": "/tmp/flowdagger_runs",
        "openpi_root": "/tmp/openpi",
        "openpi_config": "arx_connect_elevator_pins",
        "base_checkpoint": "/tmp/ckpt/name/1",
        "base_checkpoint_name": "name",
        "base_checkpoint_step": 1,
        "base_asset_id": "name",
        "raw_data_root": "/tmp/raw",
        "demo_buffer_dir": "/tmp/raw/buffer",
        "demo_buffer_count": 30,
        "addr": "tcp://*:5557",
        "default_prompt": "do the task",
        "window_stride": 10,
        "online_bc_steps": 100,
        "online_bc_batch_size": 64,
        "online_intervention_mix": 0.4,
        "online_autonomous_mix": 0.4,
        "online_demonstration_mix": 0.2,
        "steering_lr": 1e-4,
        "norm_freeze_min_success_episodes": 3,
        "inversion_batch_size": 1,
        "offline_inversion_batch_size": 2,
        "inversion_mse_threshold": 1e-3,
    }
    data.update(overrides)
    return data


@pytest.fixture
def restore_campaign_config():
    original = get_campaign_config()
    yield
    set_campaign_config(original)


def test_output_root_defaults_to_runs_root_plus_campaign_id():
    cfg = campaign_config_from_mapping(_payload())
    assert cfg.output_root == "/tmp/flowdagger_runs/arx_test_v1"


def test_explicit_output_root_wins():
    cfg = campaign_config_from_mapping(_payload(output_root="/data/custom"))
    assert cfg.output_root == "/data/custom"


def test_missing_campaign_id_raises():
    data = _payload()
    del data["campaign_id"]
    with pytest.raises(ValueError, match="missing"):
        campaign_config_from_mapping(data)


def test_load_campaign_config_from_yaml(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(_payload(campaign_id="arx_from_file")), encoding="utf-8")
    cfg = load_campaign_config(path)
    assert cfg.campaign_id == "arx_from_file"
    assert cfg.output_root.endswith("arx_from_file")
    assert cfg.source_path == str(path.resolve())


def test_set_campaign_config_is_visible_to_getters(restore_campaign_config):
    cfg = campaign_config_from_mapping(_payload(campaign_id="arx_switched"))
    set_campaign_config(cfg)
    assert get_campaign_config().campaign_id == "arx_switched"
    import arx_adapter
    assert arx_adapter.CAMPAIGN_ID == "arx_switched"
    assert arx_adapter.DEFAULT_OUTPUT_ROOT.endswith("arx_switched")
