"""Read-only live metrics snapshot for the Freesolo training dashboard.

The tracker deliberately returns a small allow-listed view. Credentials, org
identifiers, provider instance identifiers, raw prompts, and evaluation rows
never leave the server process.
"""

from __future__ import annotations

import ast
import json
import math
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .coordinator import STATE_PATH, _load_local_credentials

STAGE_ORDER = ("smoke_sft", "production_sft", "primary_grpo", "opd")
STAGE_LABELS = {
    "smoke_sft": "4B smoke SFT",
    "production_sft": "4B production SFT",
    "primary_grpo": "Primary GRPO",
    "opd": "OPD challenger",
}
TERMINAL_STATES = {"done", "failed", "cancelled", "error", "dry_run"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_METRIC_DICT_RE = re.compile(r"\{\s*'loss':\s*'[^']+'.*?\}")
_CACHE_LOCK = threading.Lock()
_CACHE: tuple[float, dict[str, Any]] | None = None


def _iso(epoch: float | int | None) -> str | None:
    if not isinstance(epoch, (int, float)):
        return None
    return datetime.fromtimestamp(float(epoch), tz=UTC).isoformat()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _worker_text(worker: dict[str, str]) -> str:
    return "\n".join(value for value in worker.values() if isinstance(value, str))


def _heartbeats(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        marker = "HEARTBEAT "
        position = line.find(marker)
        if position < 0:
            continue
        try:
            payload = json.loads(line[position + len(marker) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _printed_metrics(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    clean = _ANSI_RE.sub("", text)
    for match in _METRIC_DICT_RE.finditer(clean):
        try:
            raw = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        parsed = {
            key: number
            for key, value in raw.items()
            if (number := _finite(value)) is not None
        }
        if parsed:
            rows.append(parsed)
    return rows


def parse_worker_metrics(text: str, *, max_steps: int | None = None) -> dict[str, Any]:
    """Parse stable Flash heartbeat records without depending on terminal escape codes."""
    beats = _heartbeats(text)
    step_beats = [
        beat
        for beat in beats
        if isinstance(beat.get("step"), int) and beat.get("stage") == "sft_step"
    ]
    latest = beats[-1] if beats else {}
    latest_step = max((int(beat["step"]) for beat in step_beats), default=0)
    loss_beats = [beat for beat in step_beats if _finite(beat.get("loss")) is not None]
    printed = _printed_metrics(text)
    curve: list[dict[str, float | int]] = []
    for index, beat in enumerate(loss_beats):
        point: dict[str, float | int] = {"step": int(beat["step"])}
        for key in ("loss", "grad_norm", "learning_rate", "epoch"):
            value = _finite(beat.get(key))
            if value is not None:
                point[key] = value
        if index < len(printed):
            for key in ("entropy", "mean_token_accuracy", "num_tokens"):
                if key in printed[index]:
                    point[key] = printed[index][key]
        curve.append(point)

    first_timed = next(
        (
            beat
            for beat in step_beats
            if int(beat.get("step", 0)) > 0 and _finite(beat.get("ts")) is not None
        ),
        None,
    )
    last_timed = next(
        (
            beat
            for beat in reversed(step_beats)
            if int(beat.get("step", 0)) > 0 and _finite(beat.get("ts")) is not None
        ),
        None,
    )
    seconds_per_step = None
    eta_seconds = None
    if first_timed and last_timed and int(last_timed["step"]) > int(first_timed["step"]):
        elapsed = float(last_timed["ts"]) - float(first_timed["ts"])
        if elapsed > 0:
            seconds_per_step = elapsed / (int(last_timed["step"]) - int(first_timed["step"]))
            if max_steps:
                eta_seconds = max(0.0, max_steps - latest_step) * seconds_per_step

    gpu = latest.get("gpu") if isinstance(latest.get("gpu"), dict) else {}
    latest_loss = curve[-1] if curve else {}
    return {
        "stage": latest.get("stage"),
        "step": latest_step,
        "max_steps": max_steps,
        "progress": latest_step / max_steps if max_steps else None,
        "seconds_per_step": seconds_per_step,
        "eta_seconds": eta_seconds,
        "latest": latest_loss,
        "curve": curve,
        "gpu": {
            key: gpu.get(key)
            for key in (
                "device_name",
                "gpu_util_pct",
                "mem_util_pct",
                "memory_used_gb",
                "memory_total_gb",
                "temperature_c",
                "power_w",
                "power_limit_w",
            )
            if gpu.get(key) is not None
        },
        "heartbeat_at": _iso(_finite(latest.get("ts"))),
    }


def _safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    scalar_names = (
        "episodes",
        "turns",
        "score",
        "brier",
        "exact_operations",
        "protocol_valid",
        "validator_acceptance",
        "attack_success",
        "response_diversity",
        "repetition_rate",
    )
    safe = {name: metrics.get(name) for name in scalar_names if name in metrics}
    per_class = metrics.get("per_event_class")
    if isinstance(per_class, dict):
        safe["per_event_class"] = {
            str(name): value
            for name, value in per_class.items()
            if isinstance(value, (int, float))
        }
    return safe


def _safe_log_lines(text: str) -> list[str]:
    """Keep user-relevant lifecycle lines and omit provider infrastructure IDs."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        lowered = line.lower()
        if not line or "rented " in lowered or "instance " in lowered or "offer " in lowered:
            continue
        lines.append(line[:240])
    return lines[-12:]


def _active_stage(state: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    for name in STAGE_ORDER:
        stage = stages.get(name)
        if isinstance(stage, dict) and stage.get("state") not in TERMINAL_STATES:
            return name, stage
    for name in reversed(STAGE_ORDER):
        stage = stages.get(name)
        if isinstance(stage, dict):
            return name, stage
    return None, {}


def _stage_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    estimates = (
        state.get("cost_estimates") if isinstance(state.get("cost_estimates"), dict) else {}
    )
    estimate_keys = {
        "smoke_sft": "smoke_sft",
        "production_sft": "sft",
        "primary_grpo": "grpo",
        "opd": "opd",
    }
    cards: list[dict[str, Any]] = []
    for name in STAGE_ORDER:
        stage = stages.get(name) if isinstance(stages.get(name), dict) else {}
        estimate = estimates.get(estimate_keys[name], {})
        cards.append(
            {
                "name": name,
                "label": STAGE_LABELS[name],
                "state": stage.get("state", "pending"),
                "run_id": stage.get("run_id"),
                "cost_usd": stage.get("cost_usd", 0.0),
                "estimated_usd": estimate.get("training_usd"),
                "estimated_hours": estimate.get("wall_clock_hours"),
            }
        )
    return cards


def _snapshot_uncached() -> dict[str, Any]:
    state = _read_state()
    stage_name, local_stage = _active_stage(state)
    stages = _stage_cards(state)
    response: dict[str, Any] = {
        "generated_at": _iso(time.time()),
        "pipeline": {
            "status": state.get("status", "not_started"),
            "git_commit": state.get("git_commit"),
            "dataset_version": (state.get("dataset_manifest") or {}).get("dataset_version"),
            "environment_id": state.get("environment_id"),
            "cap_usd": (state.get("approval") or {}).get("cap_usd"),
            "prior_spend_usd": state.get("prior_spend_usd", 0.0),
            "estimated_training_usd": state.get("estimated_training_usd"),
            "evaluation_reserve_usd": state.get("evaluation_reserve_usd"),
            "combined_estimated_usd": state.get("combined_estimated_usd"),
            "stages": stages,
        },
        "active": None,
        "evaluations": {},
        "checkpoint_bank": state.get("checkpoint_bank") or {},
    }
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    response["evaluations"] = {
        name: _safe_metrics(value)
        for name, value in metrics.items()
        if isinstance(value, dict)
    }
    if not stage_name or not local_stage.get("run_id"):
        return response

    try:
        _load_local_credentials()
        from flash.client import client_from_config

        client = client_from_config()
        run_id = str(local_stage["run_id"])
        remote = client.get_run(run_id)
        spec = remote.get("spec") if isinstance(remote.get("spec"), dict) else {}
        train = spec.get("train") if isinstance(spec.get("train"), dict) else {}
        worker = client.get_worker_output(run_id)
        parsed = parse_worker_metrics(_worker_text(worker), max_steps=train.get("max_steps"))
        heartbeat = remote.get("last_heartbeat")
        if isinstance(heartbeat, dict) and not parsed["gpu"]:
            gpu = heartbeat.get("gpu")
            if isinstance(gpu, dict):
                parsed["gpu"] = {
                    key: gpu.get(key)
                    for key in (
                        "device_name",
                        "gpu_util_pct",
                        "mem_util_pct",
                        "memory_used_gb",
                        "memory_total_gb",
                        "temperature_c",
                        "power_w",
                        "power_limit_w",
                    )
                    if gpu.get(key) is not None
                }
            if not parsed["stage"]:
                parsed["stage"] = heartbeat.get("stage")
            if not parsed["heartbeat_at"]:
                parsed["heartbeat_at"] = _iso(_finite(heartbeat.get("ts")))
        logs = client.get_logs(run_id, offset=0).get("logs", "")
        recent_logs = _safe_log_lines(str(logs))
        try:
            checkpoints = client.checkpoints(run_id)
        except Exception:
            checkpoints = []
        started = (remote.get("remote") or {}).get("started_ts") or remote.get("created_at")
        elapsed = max(0.0, time.time() - float(started)) if started else None
        hourly = _finite((remote.get("remote") or {}).get("hourly_usd"))
        accrued = elapsed * hourly / 3600 if elapsed is not None and hourly else None
        response["active"] = {
            "stage_name": stage_name,
            "label": STAGE_LABELS.get(stage_name, stage_name),
            "run_id": run_id,
            "state": remote.get("state", local_stage.get("state")),
            "model": spec.get("model"),
            "algorithm": spec.get("algorithm"),
            "split": ((spec.get("environment") or {}).get("params") or {}).get("split"),
            "created_at": _iso(_finite(remote.get("created_at"))),
            "updated_at": _iso(_finite(remote.get("updated_at"))),
            "elapsed_seconds": elapsed,
            "estimated_cost_usd": remote.get("estimated_cost_usd"),
            "realized_cost_usd": remote.get("realized_cost_usd"),
            "accrued_cost_usd": accrued,
            "gpu_hourly_usd": hourly,
            "checkpoints": checkpoints,
            "recent_logs": recent_logs,
            **parsed,
        }
    except Exception as exc:
        response["active"] = {
            "stage_name": stage_name,
            "label": STAGE_LABELS.get(stage_name, stage_name),
            "run_id": local_stage.get("run_id"),
            "state": local_stage.get("state"),
            "tracker_error": str(exc),
        }
    return response


def training_snapshot(*, max_age_seconds: float = 4.0) -> dict[str, Any]:
    """Return a cached metrics snapshot to bound control-plane polling."""
    global _CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        if _CACHE is not None and now - _CACHE[0] < max_age_seconds:
            return _CACHE[1]
        snapshot = _snapshot_uncached()
        _CACHE = (now, snapshot)
        return snapshot
