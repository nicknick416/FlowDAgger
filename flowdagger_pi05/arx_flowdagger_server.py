"""OpenPI + FlowDAgger ZMQ service for ARX bimanual real-world episodes."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import cv2
import msgpack
import numpy as np
import zmq

from arx_adapter import (
    CAMERA_KEYS, PROTOCOL_VERSION,
    build_openpi_observation, is_control_hold, resolve_base_model_identity,
)
from arx_campaign import get_campaign_config, preload_campaign_config, add_arx_runtime_args
from arx_episode_store import EpisodeStore

log = logging.getLogger(__name__)
SKIP_KEYS = {
    "cmd", "state", "prompt", "extra", "episode_id", "step_id",
    "policy_version", "gripper_event", "label", "shadow_mode", "run_stage",
    "control_metrics", "task_outcome",
}
RUN_STAGES = {"demonstration", "baseline", "bootstrap", "shadow", "closed_loop", "demo"}


def validate_control_metrics(value: Any) -> dict[str, int | float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("control_metrics must be an object")
    metrics: dict[str, int | float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise ValueError("invalid control metric name")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"control metric {key!r} must be numeric")
        number = float(raw)
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"control metric {key!r} must be finite and nonnegative")
        metrics[key] = int(raw) if isinstance(raw, int) else number
    return metrics


def decode_images(message: dict[str, Any]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    raw_images = message.get("raw_images")
    if raw_images is not None:
        if not isinstance(raw_images, dict):
            raise ValueError("raw_images must be an object")
        for key in CAMERA_KEYS:
            if key not in raw_images:
                continue
            encoded = np.frombuffer(raw_images[key], dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"failed to decode raw expert image {key}")
            images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return images
    for key in CAMERA_KEYS:
        if key not in message:
            continue
        value = np.asarray(message[key], dtype=np.uint8)
        if value.ndim != 3:
            raise ValueError(f"image {key} must be 3D, got {value.shape}")
        if value.shape[0] == 3:
            value = value.transpose(1, 2, 0)
        if value.shape[-1] != 3:
            raise ValueError(f"image {key} must have 3 channels, got {value.shape}")
        images[key] = np.ascontiguousarray(value)
    return images


class PolicyRuntime(Protocol):
    policy_version: int

    def infer(self, observation: dict[str, Any], *, shadow_mode: bool) -> dict[str, Any]: ...
    def train_episode(self, episode_dir: Path, current_version: int) -> dict[str, Any]: ...


class OpenPIBaseRuntime:
    """Base-policy runtime used for record-only and protocol validation modes."""

    def __init__(
        self,
        *,
        openpi_root: str,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: str,
    ) -> None:
        self.runtime_mode = "base_record"
        root = Path(openpi_root)
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root / "packages" / "openpi-client" / "src"))
        from openpi.policies import policy_config
        from openpi.training import config

        self.default_prompt = default_prompt
        self.base_identity = resolve_base_model_identity(checkpoint_dir)
        self.base_model_id = self.base_identity["base_model_id"]
        self.steering_eligible = False
        train_config = config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
        )
        self.policy_version = 0

    def infer(self, observation: dict[str, Any], *, shadow_mode: bool) -> dict[str, Any]:
        result = self.policy.infer(observation)
        return {
            "actions": np.asarray(result["actions"], dtype=np.float32),
            "model_infer_ms": float(result.get("policy_timing", {}).get("infer_ms", 0.0)),
            "executed_policy": "base",
            "shadow_mode": bool(shadow_mode),
        }

    def train_episode(self, episode_dir: Path, current_version: int) -> dict[str, Any]:
        return {
            "skipped": True,
            "reason": "record-only base runtime",
            "policy_version": current_version,
        }


class ProtocolOnlyRuntime:
    """GPU-free runtime for transport/state-machine validation only."""

    def __init__(self, *, policy_version: int = 0) -> None:
        self.runtime_mode = "protocol_only"
        self.policy_version = int(policy_version)
        self.base_model_id = "protocol-only"
        self.steering_eligible = False

    def infer(self, observation: dict[str, Any], *, shadow_mode: bool) -> dict[str, Any]:
        return {
            "actions": np.zeros((50, 20), dtype=np.float32),
            "model_infer_ms": 0.0,
            "executed_policy": "protocol_only",
        }

    def train_episode(self, episode_dir: Path, current_version: int) -> dict[str, Any]:
        raise RuntimeError("protocol-only runtime cannot train")


class TrainingCoordinator:
    """Run episode-boundary training without blocking the REP socket."""

    def __init__(self, runtime: PolicyRuntime) -> None:
        self.runtime = runtime
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "state": "idle",
            "policy_version": int(runtime.policy_version),
            "metrics": {},
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def submit(self, episode_dir: Path) -> None:
        with self._lock:
            if self._status["state"] == "running":
                raise RuntimeError("training is already running")
            version = int(self._status["policy_version"])
            self._status = {
                "state": "running",
                "episode_dir": str(episode_dir),
                "policy_version": version,
                "metrics": {},
            }

        def worker() -> None:
            try:
                metrics = self.runtime.train_episode(episode_dir, version)
                new_version = int(metrics.get("policy_version", self.runtime.policy_version))
                result_state = str(metrics.get("state", "succeeded"))
                if result_state not in ("succeeded", "no_improvement", "rejected"):
                    result_state = "succeeded"
                with self._lock:
                    self._status = {
                        "state": result_state,
                        "episode_dir": str(episode_dir),
                        "policy_version": new_version,
                        "metrics": metrics,
                    }
            except Exception as exc:
                log.exception("FlowDAgger episode training failed")
                with self._lock:
                    self._status = {
                        "state": "failed",
                        "episode_dir": str(episode_dir),
                        "policy_version": version,
                        "error": repr(exc),
                        "metrics": {},
                    }

        threading.Thread(target=worker, daemon=True, name="flowdagger-trainer").start()

    def acknowledge(self) -> None:
        with self._lock:
            if self._status["state"] in (
                "succeeded", "no_improvement", "rejected", "failed"
            ):
                self._status = {
                    "state": "idle",
                    "policy_version": int(self.runtime.policy_version),
                    "metrics": self._status.get("metrics", {}),
                }


class ARXFlowDaggerServer:
    def __init__(
        self,
        runtime: PolicyRuntime,
        *,
        output_root: str | Path,
        addr: str = "tcp://*:5556",
        default_prompt: str,
    ) -> None:
        self.runtime = runtime
        self.output_root = Path(output_root)
        self.store = EpisodeStore(output_root)
        self.trainer = TrainingCoordinator(runtime)
        self.addr = addr
        self.default_prompt = default_prompt
        self._episode_policy_version: int | None = None
        self._shadow_mode = False
        self._run_stage = "demonstration"
        self._intervening = False
        self._demo_active = False
        self._server_session_id = uuid.uuid4().hex
        self._client_session_id: str | None = None
        self._last_heartbeat_monotonic: float | None = None
        self._strict_protocol = str(
            getattr(self.runtime, "base_model_id", "unknown")
        ) != "unknown"
        self._reset_intervention_pause_state()

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        cmd = message.get("cmd", "predict")
        if cmd == "health":
            return {
                "status": "ok",
                "runtime_mode": str(getattr(self.runtime, "runtime_mode", "unknown")),
                "protocol_version": PROTOCOL_VERSION,
                "server_session_id": self._server_session_id,
                "base_model_id": str(getattr(self.runtime, "base_model_id", "unknown")),
                "campaign_id": get_campaign_config().campaign_id,
                "bootstrap_reviewed_episodes": int(
                    getattr(self.runtime, "bootstrap_reviewed_episodes", 0)
                ),
                "steering_eligible": bool(getattr(self.runtime, "steering_eligible", False)),
                "policy_version": int(self.runtime.policy_version),
                "active_episode_id": self.store.active_episode_id,
                "training": self.trainer.snapshot(),
                "action_horizon": 50,
                "action_dim": 20,
                "state_dim": 20,
                "camera_keys": list(CAMERA_KEYS),
                "stage_counts": self._stage_counts(),
                "heartbeat_age_s": (
                    None if self._last_heartbeat_monotonic is None
                    else max(0.0, time.monotonic() - self._last_heartbeat_monotonic)
                ),
            }
        if cmd == "episode_start":
            if self._demo_active:
                active = self.output_root / "steering_checkpoints" / "ACTIVE"
                if not active.is_file():
                    raise RuntimeError("cannot leave demo: no ACTIVE steering checkpoint")
                self.runtime.load_version(int(active.read_text(encoding="utf-8").strip()))
                self._demo_active = False
            runtime_base = str(getattr(self.runtime, "base_model_id", "unknown"))
            # Lightweight test runtimes have no deployment identity. Real
            # runtimes always take the strict protocol-v2 path.
            strict_deployment = runtime_base != "unknown"
            if strict_deployment and int(message.get("protocol_version", -1)) != PROTOCOL_VERSION:
                raise RuntimeError(f"protocol_version must be {PROTOCOL_VERSION}")
            client_session_id = str(message.get("client_session_id", "legacy-test"))
            if not client_session_id:
                raise ValueError("client_session_id is required")
            if strict_deployment and str(message.get("base_model_id", "")) != runtime_base:
                raise RuntimeError(
                    f"base model mismatch: server={runtime_base}, "
                    f"client={message.get('base_model_id')}"
                )
            if (
                strict_deployment
                and str(message.get("server_session_id", ""))
                != self._server_session_id
            ):
                raise RuntimeError("server session mismatch during episode_start")
            episode_id = str(message["episode_id"])
            status = self.trainer.snapshot()
            if status["state"] == "running":
                raise RuntimeError("cannot start an episode while training is running")
            self.trainer.acknowledge()
            self._episode_policy_version = int(self.runtime.policy_version)
            requested_policy = int(message.get(
                "requested_policy_version", self._episode_policy_version
            ))
            if requested_policy != self._episode_policy_version:
                raise RuntimeError(
                    "requested policy version mismatch: "
                    f"server={self._episode_policy_version}, client={requested_policy}"
                )
            self._shadow_mode = bool(message.get("shadow_mode", False))
            self._run_stage = str(message.get("run_stage", "demonstration"))
            if self._run_stage not in RUN_STAGES:
                raise ValueError(f"invalid run_stage: {self._run_stage}")
            runtime_mode = str(getattr(self.runtime, "runtime_mode", "unknown"))
            if runtime_mode == "flowdagger" and self._run_stage in (
                "demonstration", "baseline"
            ):
                raise RuntimeError(
                    f"run_stage={self._run_stage} requires --record-only server"
                )
            if runtime_mode == "base_record" and self._run_stage in (
                "shadow", "closed_loop"
            ):
                raise RuntimeError(
                    f"run_stage={self._run_stage} requires full FlowDAgger server"
                )
            if self._run_stage in ("shadow", "closed_loop") and self._episode_policy_version <= 0:
                raise RuntimeError(
                    f"run_stage={self._run_stage} requires a trained steering checkpoint"
                )
            if self._run_stage == "closed_loop" and self._shadow_mode:
                raise RuntimeError("closed_loop requires shadow_mode=false")
            if self._run_stage != "closed_loop" and not self._shadow_mode:
                raise RuntimeError(
                    f"run_stage={self._run_stage} requires shadow_mode=true"
                )
            self._intervening = False
            self._client_session_id = client_session_id
            self._last_heartbeat_monotonic = time.monotonic()
            self._reset_intervention_pause_state()
            self.store.start(
                episode_id,
                prompt=str(message.get("prompt") or self.default_prompt),
                base_policy_version=0,
                steering_policy_version=self._episode_policy_version,
                shadow_mode=self._shadow_mode,
                run_stage=self._run_stage,
                runtime_mode=str(getattr(self.runtime, "runtime_mode", "unknown")),
                protocol_version=PROTOCOL_VERSION,
                client_session_id=client_session_id,
                server_session_id=self._server_session_id,
                base_model_id=runtime_base,
            )
            return {
                "status": "ok",
                "policy_version": self._episode_policy_version,
                "server_session_id": self._server_session_id,
                "base_model_id": runtime_base,
            }

        if cmd == "demo_start":
            if self.store.active_episode_id is not None:
                raise RuntimeError("cannot start demo while an episode is active")
            if self.trainer.snapshot()["state"] == "running":
                raise RuntimeError("cannot start demo while training is running")
            if str(getattr(self.runtime, "runtime_mode", "unknown")) != "flowdagger":
                raise RuntimeError("demo requires the full FlowDAgger server")
            requested = message.get("steering_version", "active")
            if str(requested).lower() == "active":
                active = self.output_root / "steering_checkpoints" / "ACTIVE"
                if not active.is_file():
                    raise RuntimeError("no ACTIVE steering checkpoint")
                version = int(active.read_text(encoding="utf-8").strip())
            else:
                version = int(requested)
            version = int(self.runtime.load_version(version))
            self._demo_active = True
            self._shadow_mode = False
            self._run_stage = "demo"
            return {
                "status": "ok",
                "policy_version": version,
                "recording": False,
                "training": False,
            }

        if cmd == "demo_stop":
            active = self.output_root / "steering_checkpoints" / "ACTIVE"
            if active.is_file():
                self.runtime.load_version(int(active.read_text(encoding="utf-8").strip()))
            self._demo_active = False
            return {"status": "ok", "policy_version": int(self.runtime.policy_version)}

        if cmd == "control_heartbeat":
            self._require_episode(message)
            self._last_heartbeat_monotonic = time.monotonic()
            return {
                "status": "ok",
                "server_session_id": self._server_session_id,
                "policy_version": self._episode_policy_version,
            }

        if cmd in ("intervention_start", "intervention_stop"):
            self._require_episode(message)
            self._intervening = cmd == "intervention_start"
            if self._intervening:
                self._reset_intervention_pause_state(keep_last_state=True)
                self.store.append_event(
                    cmd,
                    step_id=int(message.get("step_id", 0)),
                )
                if "state" in message:
                    images = decode_images(message)
                    boundary_record_sequence = self.store.append_observation(
                        kind="boundary",
                        step_id=int(message["step_id"]),
                        images=images,
                        state=message["state"],
                        prompt=str(message.get("prompt") or self.default_prompt),
                        request_generation=int(message.get("request_generation", 0)),
                    )
                    self._last_control_state = message["state"]
                    self.store.append_event(
                        "intervention_boundary_saved",
                        boundary_record_sequence=boundary_record_sequence,
                    )
            else:
                dropped = self._drop_trailing_holds()
                self.store.append_event(cmd, dropped_trailing_holds=dropped)
            return {"status": "ok", "intervening": self._intervening}

        if cmd == "expert_step":
            self._require_episode(message)
            if not self._intervening:
                raise RuntimeError("expert_step requires an active intervention")
            return self._record_expert_step(message)

        if cmd == "predict":
            request_started = time.monotonic()
            if self.store.active_episode_id is not None:
                if self._strict_protocol:
                    missing = [
                        key for key in (
                            "step_id", "policy_version", "request_generation",
                            "client_session_id", "server_session_id",
                        ) if key not in message
                    ]
                    if missing:
                        raise RuntimeError(
                            f"predict is missing protocol-v2 fields: {missing}"
                        )
                self._require_episode(message)
                # Valid policy traffic is the control heartbeat. This avoids a
                # second concurrent REQ call from the real-time client.
                self._last_heartbeat_monotonic = time.monotonic()
            images = decode_images(message)
            prompt = str(message.get("prompt") or self.default_prompt)
            observation = build_openpi_observation(images, message.get("state", []), prompt)
            result = self.runtime.infer(observation, shadow_mode=self._shadow_mode)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (50, 20):
                raise ValueError(f"policy must return (50,20), got {actions.shape}")
            if self.store.active_episode_id is not None:
                shadow_actions = result.get("shadow_actions")
                shadow_fields: dict[str, Any] = {}
                if shadow_actions is not None:
                    shadow = np.asarray(shadow_actions, dtype=np.float32)
                    difference = shadow - actions
                    shadow_fields = {
                        "shadow_action_mse": float(np.mean(np.square(difference))),
                        "shadow_action_max_abs": float(np.max(np.abs(difference))),
                    }
                self.store.append_observation(
                    kind="policy",
                    step_id=int(message.get("step_id", 0)),
                    images=images,
                    state=message.get("state", []),
                    prompt=prompt,
                    policy_version=self._episode_policy_version,
                    executed_policy=result.get("executed_policy", "base"),
                    predicted_actions=actions.tolist(),
                    **shadow_fields,
                )
                self._last_control_state = message.get("state", [])
            return {
                "status": "ok",
                "actions": actions.tolist(),
                "policy_version": self._episode_policy_version or int(self.runtime.policy_version),
                "model_infer_time_ms": float(result.get("model_infer_ms", 0.0)),
                "executed_policy": result.get("executed_policy", "base"),
                "episode_id": message.get("episode_id"),
                "step_id": int(message.get("step_id", 0)),
                "request_generation": int(message.get("request_generation", 0)),
                "server_session_id": self._server_session_id,
                "response_time_ms": (time.monotonic() - request_started) * 1000.0,
            }

        if cmd == "episode_end":
            self._require_episode(message)
            self._drop_trailing_holds()
            task_outcome = message.get("task_outcome")
            if task_outcome is None:
                legacy_label = str(message.get("label", ""))
                task_outcome = (
                    "success" if legacy_label.endswith("success") else legacy_label
                )
            task_outcome = str(task_outcome)
            active_summary = self.store.active_summary
            expert_transitions = int(
                active_summary.get("expert_transitions", 0)
            )
            episode_dir = self.store.finish(
                task_outcome,
                control_metrics=validate_control_metrics(
                    message.get("control_metrics")
                ),
            )
            self._intervening = False
            self._reset_intervention_pause_state()
            # Collection/evaluation stages must not mutate the active policy.
            # Online DAgger updates happen only at closed-loop episode boundaries.
            training_queued = (
                task_outcome == "success"
                and int(active_summary.get("intervention_count", 0)) > 0
                and self._run_stage in ("bootstrap", "closed_loop")
                and expert_transitions >= 1
            )
            if training_queued:
                self.trainer.submit(episode_dir)
            self._client_session_id = None
            self._last_heartbeat_monotonic = None
            return {
                "status": "ok",
                "task_outcome": task_outcome,
                "training_queued": training_queued,
                "episode_dir": str(episode_dir),
            }

        if cmd == "train_status":
            return {"status": "ok", **self.trainer.snapshot()}
        if cmd == "reset":
            return {"status": "ok"}
        raise ValueError(f"unknown command: {cmd}")

    def _reset_intervention_pause_state(self, *, keep_last_state: bool = False) -> None:
        self._expert_hold_buffer: list[dict[str, Any]] = []
        self._intervention_motion_started = False
        if not keep_last_state:
            self._last_control_state = None

    def _drop_trailing_holds(self) -> int:
        dropped = len(self._expert_hold_buffer)
        self._expert_hold_buffer.clear()
        return dropped

    def _write_expert_observation(self, payload: dict[str, Any]) -> None:
        self.store.append_observation(
            kind="expert",
            step_id=int(payload["step_id"]),
            images=payload["images"],
            state=payload["state"],
            prompt=payload["prompt"],
            gripper_event=payload.get("gripper_event"),
            timestamp_s=payload.get("timestamp_s"),
        )
        self._last_control_state = payload["state"]

    def _record_expert_step(self, message: dict[str, Any]) -> dict[str, Any]:
        images = decode_images(message)
        state = message.get("state", [])
        payload = {
            "step_id": int(message["step_id"]),
            "images": images,
            "state": state,
            "prompt": str(message.get("prompt") or self.default_prompt),
            "gripper_event": message.get("gripper_event"),
            "timestamp_s": time.time(),
        }
        hold = (
            self._last_control_state is not None
            and is_control_hold(self._last_control_state, state)
        )
        if not self._intervention_motion_started:
            if hold:
                return {"status": "ok", "skipped_pause": True}
            self._intervention_motion_started = True
            self._write_expert_observation(payload)
            return {"status": "ok"}
        if hold:
            self._expert_hold_buffer.append(payload)
            self._last_control_state = state
            return {"status": "ok", "buffered_hold": True}
        for held in self._expert_hold_buffer:
            self._write_expert_observation(held)
        self._expert_hold_buffer.clear()
        self._write_expert_observation(payload)
        return {"status": "ok"}

    def handle_raw(self, raw: bytes) -> bytes:
        try:
            message = msgpack.unpackb(raw, raw=False)
            response = self.handle_message(message)
        except Exception as exc:
            log.exception("request failed")
            response = {"status": "error", "message": str(exc)}
        return msgpack.packb(response)

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.addr)
        log.info("ARX FlowDAgger server listening on %s", self.addr)
        try:
            while True:
                socket.send(self.handle_raw(socket.recv()))
        finally:
            self.store.abort_incomplete("server_shutdown")
            socket.close()
            context.term()

    def _require_episode(self, message: dict[str, Any]) -> None:
        episode_id = str(message.get("episode_id", ""))
        if not episode_id or episode_id != self.store.active_episode_id:
            raise RuntimeError(
                f"episode mismatch: active={self.store.active_episode_id!r}, request={episode_id!r}"
            )
        if "policy_version" in message and self._episode_policy_version is not None:
            requested_version = int(message["policy_version"])
            if requested_version != self._episode_policy_version:
                raise RuntimeError(
                    "policy version mismatch: "
                    f"episode={self._episode_policy_version}, request={requested_version}"
                )
        if self._client_session_id is not None:
            if self._strict_protocol and "client_session_id" not in message:
                raise RuntimeError("client_session_id is required")
            if "client_session_id" in message and str(
                message["client_session_id"]
            ) != self._client_session_id:
                raise RuntimeError("client session mismatch")
        if self._strict_protocol and "server_session_id" not in message:
            raise RuntimeError("server_session_id is required")
        if (
            "server_session_id" in message
            and str(message["server_session_id"]) != self._server_session_id
        ):
            raise RuntimeError("server session mismatch")

    def _stage_counts(self) -> dict[str, dict[str, int]]:
        counts = {
            stage: {"total": 0, "success": 0}
            for stage in RUN_STAGES
        }
        for directory in self.store.episodes_root.glob("*"):
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("runtime_mode") == "protocol_only":
                    continue
                steps_path = directory / "steps.jsonl"
                if steps_path.is_file() and any(
                    json.loads(line).get("executed_policy") == "protocol_only"
                    for line in steps_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ):
                    continue
                stage = metadata.get("run_stage")
                if stage not in counts:
                    continue
                counts[stage]["total"] += 1
                if metadata.get("label") in (
                    "assisted_success", "autonomous_success"
                ):
                    counts[stage]["success"] += 1
            except (OSError, ValueError, TypeError):
                log.warning("ignoring unreadable episode metadata: %s", metadata_path)
        return counts


def main() -> None:
    cfg = preload_campaign_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default=cfg.addr)
    add_arx_runtime_args(parser, cfg)
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="GPU-free ZMQ/state validation; never use for robot control",
    )
    parser.add_argument("--protocol-policy-version", type=int, default=0)
    args = parser.parse_args()
    if args.record_only and args.protocol_only:
        parser.error("--record-only and --protocol-only are mutually exclusive")
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "service.log", encoding="utf-8"),
        ],
    )
    log.info(
        "loaded campaign config %s campaign_id=%s output_root=%s",
        cfg.source_path,
        cfg.campaign_id,
        args.output_root,
    )
    if args.protocol_only:
        log.warning("PROTOCOL-ONLY MODE: returned actions are zeros")
        runtime = ProtocolOnlyRuntime(policy_version=args.protocol_policy_version)
    elif args.record_only:
        runtime: PolicyRuntime = OpenPIBaseRuntime(
            openpi_root=args.openpi_root,
            config_name=args.config,
            checkpoint_dir=args.checkpoint,
            default_prompt=args.default_prompt,
        )
    else:
        from arx_trainer import ARXFlowDaggerRuntime
        runtime = ARXFlowDaggerRuntime(
            openpi_root=args.openpi_root,
            config_name=args.config,
            checkpoint_dir=args.checkpoint,
            output_root=args.output_root,
            default_prompt=args.default_prompt,
        )
    ARXFlowDaggerServer(
        runtime,
        output_root=args.output_root,
        addr=args.addr,
        default_prompt=args.default_prompt,
    ).run()


if __name__ == "__main__":
    main()
