"""Load ARX FlowDAgger campaign settings from YAML instead of hardcoded paths."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "arx_campaign.yaml"

_REQUIRED_STR = (
    "campaign_id",
    "runs_root",
    "openpi_root",
    "openpi_config",
    "base_checkpoint",
    "base_checkpoint_name",
    "base_asset_id",
    "raw_data_root",
    "demo_buffer_dir",
    "addr",
    "default_prompt",
)


@dataclass(frozen=True, slots=True)
class ARXCampaignConfig:
    campaign_id: str
    runs_root: str
    output_root: str
    openpi_root: str
    openpi_config: str
    base_checkpoint: str
    base_checkpoint_name: str
    base_checkpoint_step: int
    base_asset_id: str
    raw_data_root: str
    demo_buffer_dir: str
    demo_buffer_count: int
    addr: str
    default_prompt: str
    window_stride: int
    online_bc_steps: int
    online_bc_batch_size: int
    online_intervention_mix: float
    online_autonomous_mix: float
    online_demonstration_mix: float
    steering_lr: float
    norm_freeze_min_success_episodes: int
    inversion_batch_size: int
    offline_inversion_batch_size: int
    inversion_mse_threshold: float
    source_path: str


_current: ARXCampaignConfig | None = None


def default_config_path() -> Path:
    env = os.environ.get("FLOWDAGGER_CONFIG")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_PATH


def load_campaign_config(path: str | Path | None = None) -> ARXCampaignConfig:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"campaign config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"campaign config must be a mapping: {config_path}")
    return campaign_config_from_mapping(payload, source_path=config_path)


def campaign_config_from_mapping(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ARXCampaignConfig:
    missing = [key for key in _REQUIRED_STR if not str(data.get(key) or "").strip()]
    if missing:
        raise ValueError(f"campaign config missing {missing}")
    campaign_id = str(data["campaign_id"]).strip()
    runs_root = _as_path_string(data["runs_root"])
    output_root = data.get("output_root")
    if output_root in (None, ""):
        output_root = str(Path(runs_root) / campaign_id)
    else:
        output_root = _as_path_string(output_root)
    mix = (
        _as_float(data, "online_intervention_mix"),
        _as_float(data, "online_autonomous_mix"),
        _as_float(data, "online_demonstration_mix"),
    )
    if any(value < 0 for value in mix) or sum(mix) <= 0:
        raise ValueError("online mix weights must be nonnegative and not all zero")
    inversion_batch_size = _as_int(data, "inversion_batch_size")
    offline_inversion_batch_size = _as_int(data, "offline_inversion_batch_size")
    for name, value in (
        ("inversion_batch_size", inversion_batch_size),
        ("offline_inversion_batch_size", offline_inversion_batch_size),
    ):
        if not 1 <= value <= 4:
            raise ValueError(f"{name} must be in [1, 4]")
    steering_lr = _as_float(data, "steering_lr")
    if not steering_lr > 0:
        raise ValueError("steering_lr must be positive")
    return ARXCampaignConfig(
        campaign_id=campaign_id,
        runs_root=runs_root,
        output_root=output_root,
        openpi_root=_as_path_string(data["openpi_root"]),
        openpi_config=str(data["openpi_config"]).strip(),
        base_checkpoint=_as_path_string(data["base_checkpoint"]),
        base_checkpoint_name=str(data["base_checkpoint_name"]).strip(),
        base_checkpoint_step=_as_int(data, "base_checkpoint_step"),
        base_asset_id=str(data["base_asset_id"]).strip(),
        raw_data_root=_as_path_string(data["raw_data_root"]),
        demo_buffer_dir=_as_path_string(data["demo_buffer_dir"]),
        demo_buffer_count=_as_int(data, "demo_buffer_count"),
        addr=str(data["addr"]).strip(),
        default_prompt=str(data["default_prompt"]).strip(),
        window_stride=_as_int(data, "window_stride"),
        online_bc_steps=_as_int(data, "online_bc_steps"),
        online_bc_batch_size=_as_int(data, "online_bc_batch_size"),
        online_intervention_mix=mix[0],
        online_autonomous_mix=mix[1],
        online_demonstration_mix=mix[2],
        steering_lr=steering_lr,
        norm_freeze_min_success_episodes=_as_int(
            data, "norm_freeze_min_success_episodes"
        ),
        inversion_batch_size=inversion_batch_size,
        offline_inversion_batch_size=offline_inversion_batch_size,
        inversion_mse_threshold=_as_float(data, "inversion_mse_threshold"),
        source_path=str(Path(source_path).resolve()) if source_path else "",
    )


def get_campaign_config() -> ARXCampaignConfig:
    global _current
    if _current is None:
        set_campaign_config(load_campaign_config())
    return _current


def set_campaign_config(cfg: ARXCampaignConfig) -> ARXCampaignConfig:
    global _current
    _current = cfg
    _publish_path_aliases(sys.modules[__name__], cfg)
    adapter = sys.modules.get("arx_adapter")
    if adapter is not None:
        _publish_path_aliases(adapter, cfg)
    trainer = sys.modules.get("arx_trainer")
    if trainer is not None:
        trainer.ONLINE_BC_STEPS = cfg.online_bc_steps
        trainer.ONLINE_BC_BATCH_SIZE = cfg.online_bc_batch_size
        trainer.ONLINE_INTERVENTION_MIX = cfg.online_intervention_mix
        trainer.ONLINE_AUTONOMOUS_MIX = cfg.online_autonomous_mix
        trainer.ONLINE_DEMONSTRATION_MIX = cfg.online_demonstration_mix
        trainer.NORM_FREEZE_MIN_SUCCESS_EPISODES = cfg.norm_freeze_min_success_episodes
    return cfg


def preload_campaign_config(argv: list[str] | None = None) -> ARXCampaignConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-file", default=None)
    known, _ = parser.parse_known_args(argv)
    return set_campaign_config(load_campaign_config(known.config_file))


def add_config_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-file",
        default=str(default_config_path()),
        help="ARX campaign YAML (paths, prompt, training knobs)",
    )


def add_arx_runtime_args(parser: argparse.ArgumentParser, cfg: ARXCampaignConfig) -> None:
    add_config_file_arg(parser)
    parser.add_argument("--output-root", default=cfg.output_root)
    parser.add_argument("--openpi-root", default=cfg.openpi_root)
    parser.add_argument("--config", default=cfg.openpi_config)
    parser.add_argument("--checkpoint", default=cfg.base_checkpoint)
    parser.add_argument("--default-prompt", default=cfg.default_prompt)


def _publish_path_aliases(module: Any, cfg: ARXCampaignConfig) -> None:
    module.CAMPAIGN_ID = cfg.campaign_id
    module.DEFAULT_OUTPUT_ROOT = cfg.output_root
    module.BASE_CHECKPOINT = cfg.base_checkpoint
    module.BASE_CHECKPOINT_NAME = cfg.base_checkpoint_name
    module.BASE_CHECKPOINT_STEP = cfg.base_checkpoint_step
    module.BASE_ASSET_ID = cfg.base_asset_id
    module.RAW_DATA_ROOT = cfg.raw_data_root
    module.DEMO_BUFFER_DIR = cfg.demo_buffer_dir
    module.DEMO_BUFFER_COUNT = cfg.demo_buffer_count
    module.WINDOW_STRIDE = cfg.window_stride


def _as_path_string(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("path value cannot be empty")
    return str(Path(text).expanduser())


def _as_int(data: Mapping[str, Any], key: str) -> int:
    if key not in data:
        raise ValueError(f"campaign config missing {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"campaign config {key!r} must be an integer")
    number = int(value)
    if float(value) != number:
        raise ValueError(f"campaign config {key!r} must be an integer")
    if number <= 0:
        raise ValueError(f"campaign config {key!r} must be positive")
    return number


def _as_float(data: Mapping[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"campaign config missing {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"campaign config {key!r} must be a number")
    number = float(value)
    if number != number:  # NaN
        raise ValueError(f"campaign config {key!r} must be finite")
    return number


CAMPAIGN_ID = ""
DEFAULT_OUTPUT_ROOT = ""
BASE_CHECKPOINT = ""
BASE_CHECKPOINT_NAME = ""
BASE_CHECKPOINT_STEP = 0
BASE_ASSET_ID = ""
RAW_DATA_ROOT = ""
DEMO_BUFFER_DIR = ""
DEMO_BUFFER_COUNT = 0
WINDOW_STRIDE = 0
