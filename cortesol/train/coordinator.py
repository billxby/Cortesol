"""Resumable, cost-gated coordinator for the complete Cortesol training lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bundle import build_bundle
from .config_artifacts import render_configs
from .datasets import build_all
from .evaluate import evaluate_rows, flash_responder, load_rows, prior_brier, promotion_gate

ROOT = Path(__file__).parents[2]
RUN_ROOT = ROOT / "runs" / "training"
STATE_PATH = RUN_ROOT / "state.json"
DATA_DIR = RUN_ROOT / "data"
BUNDLE_DIR = RUN_ROOT / "environment"
RESOLVED_CONFIG_DIR = RUN_ROOT / "configs"
STAGES = ("smoke_sft", "production_sft", "primary_grpo", "opd", "final")
TERMINAL_STATES = {"done", "failed", "cancelled", "error", "dry_run"}


def _load_local_credentials() -> None:
    """Load only the Freesolo key from ignored dotenv files without executing them."""
    if os.environ.get("FREESOLO_API_KEY"):
        return
    for path in (ROOT / ".env.local", ROOT / ".env"):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != "FREESOLO_API_KEY":
                continue
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ["FREESOLO_API_KEY"] = value
                return


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return result


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"schema_version": 1, "stages": {}, "deployments": {}, "metrics": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_conda() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "cortesol-train":
        raise RuntimeError(
            "activate the environment first: conda activate cortesol-train "
            "(or use conda run -n cortesol-train)"
        )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"cortesol-train must use Python 3.12, got {sys.version.split()[0]}")


def _verify_branch(*, fetch: bool) -> str:
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    if branch != "feature/training":
        raise RuntimeError(f"training pipeline must run on feature/training, not {branch!r}")
    if fetch:
        _run(["git", "fetch", "origin", "main", "feature/training"])
    behind = int(_run(["git", "rev-list", "--count", "HEAD..origin/main"]).stdout.strip())
    if behind:
        raise RuntimeError(f"feature/training is behind current origin/main by {behind} commit(s)")
    return _run(["git", "rev-parse", "HEAD"]).stdout.strip()


def _verify_configs(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    from flash.schema import spec_from_file

    specs: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        spec = spec_from_file(str(path))
        if name in {"grpo", "opd", "grpo_opd"} and not spec.train.structured_outputs:
            raise RuntimeError(f"{name} lost schema-constrained decoding")
        if name in {"grpo", "opd", "grpo_opd"}:
            raw = json.loads(spec.train.structured_outputs)
            schema_file = json.loads((RESOLVED_CONFIG_DIR / "ops.schema.json").read_text())
            if raw.get("json") != schema_file:
                raise RuntimeError(
                    f"{name} structured output schema drifted from ops_json_schema()"
                )
        specs[name] = spec.to_dict()
    return specs


def _cost_estimates(paths: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], float]:
    from flash.cost import estimate_cost, runconfig_from_spec
    from flash.schema import spec_from_file

    estimates: dict[str, dict[str, Any]] = {}
    total = 0.0
    for name, path in paths.items():
        estimate = estimate_cost(runconfig_from_spec(spec_from_file(str(path))))
        estimates[name] = {
            "training_usd": estimate.total_usd,
            "managed_teacher_usd": estimate.teacher_api_usd,
            "gpu": estimate.gpu,
            "wall_clock_hours": estimate.wall_clock_hours,
        }
        # GRPO-from-OPD remains a checked optional config, but the anytime
        # default is the shorter SFT -> GRPO -> OPD ladder.
        if name != "grpo_opd":
            total += estimate.total_usd
    return estimates, round(total, 4)


def _evaluation_reserve(manifest: dict[str, Any]) -> float:
    """Conservative serving charge for every planned checkpoint evaluation.

    Reserve each turn for all prior user contexts and maximally long model
    completions. This is stricter than replaying gold-length outputs, while
    avoiding the old assumption that even the first turn already contained a
    full 12,288-token transcript.
    """
    from flash.serve.pricing import serving_price

    files = manifest["files"]
    max_user_tokens = max(int(item["input_tokens"]["max"]) for item in files.values())
    contract_tokens = (len((ROOT / "cortesol/train/TRAINING_CONTRACT.md").read_text()) + 3) // 4
    completion_tokens = 256

    # 4B anytime ladder: every development checkpoint (3 SFT + 4 GRPO + 3 OPD),
    # one security evaluation per promoted stage, and both sealed-final candidates.
    development_evaluations = 128 * (3 + 4 + 3)
    security_evaluations = 5 * 3
    final_evaluations = 256 * 2
    episodes = {
        "Qwen/Qwen3.5-4B": (
            1 + development_evaluations + security_evaluations + final_evaluations
        )
    }
    input_tokens_per_episode = sum(
        contract_tokens
        + turn * max_user_tokens
        + (turn - 1) * completion_tokens
        for turn in range(1, 25)
    )
    reserve = 0.0
    for model, episode_count in episodes.items():
        price = serving_price(model)
        calls = episode_count * 24
        reserve += (
            episode_count * input_tokens_per_episode * price.billed_input_usd_per_mtok
            + calls * completion_tokens * price.billed_output_usd_per_mtok
        ) / 1_000_000
    return round(reserve, 4)


def _fingerprint(state: dict[str, Any]) -> str:
    payload = {
        "git_commit": state["git_commit"],
        "bundle_sha256": state["bundle_sha256"],
        "combined_estimated_usd": state["combined_estimated_usd"],
        "environment_id": state.get("environment_id"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _published_environment(bundle_dir: Path, name: str) -> str:
    result = _run(["flash", "env", "push", "--name", name, str(bundle_dir)])
    for line in result.stdout.splitlines():
        if line.startswith("published "):
            return line.removeprefix("published ").strip()
    raise RuntimeError(f"could not parse published environment id: {result.stdout.strip()}")


def _server_dry_run(path: Path) -> dict[str, Any]:
    result = _run(["flash", "train", str(path), "--dry-run"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Flash dry-run returned non-JSON output: {result.stdout}") from exc


def _quote_usd(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key in ("cost_usd", "estimated_cost_usd", "total_usd"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for value in payload.values():
            found = _quote_usd(value)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _quote_usd(value)
            if found is not None:
                return found
    return None


def preflight(*, environment_name: str, fetch: bool, local_only: bool) -> dict[str, Any]:
    _load_local_credentials()
    _assert_conda()
    commit = _verify_branch(fetch=fetch)
    _run([sys.executable, "-m", "pytest"])
    _run([sys.executable, "-m", "ruff", "check", "cortesol", "tests"])
    manifest = build_all(DATA_DIR)
    bundle = build_bundle(BUNDLE_DIR, data_dir=DATA_DIR, source_root=ROOT)
    state = _load_state()
    lineage_changed = any(
        (
            state.get("git_commit") and state.get("git_commit") != commit,
            state.get("bundle_sha256") and state.get("bundle_sha256") != bundle["sha256"],
        )
    )
    if lineage_changed and state.get("stages"):
        prior_stages = dict(state["stages"])
        prior_spend = sum(float(stage.get("cost_usd") or 0.0) for stage in prior_stages.values())
        state.setdefault("lineage_history", []).append(
            {
                "git_commit": state.get("git_commit"),
                "bundle_sha256": state.get("bundle_sha256"),
                "stages": prior_stages,
                "spend_usd": prior_spend,
            }
        )
        state["prior_spend_usd"] = round(
            float(state.get("prior_spend_usd") or 0.0) + prior_spend, 4
        )
        state["stages"] = {}
        state["deployments"] = {}
        state["metrics"] = {}
        for key in (
            "approval",
            "approval_fingerprint",
            "winner",
            "sealed_final_evaluated",
        ):
            state.pop(key, None)
    state.update(
        {
            "git_commit": commit,
            "dataset_manifest": manifest,
            "bundle_sha256": bundle["sha256"],
            "status": "local_preflight_complete",
        }
    )
    placeholder_paths = render_configs(
        RESOLVED_CONFIG_DIR, environment_id="local/cortesol-belief-update"
    )
    state["config_specs"] = _verify_configs(placeholder_paths)
    estimates, total = _cost_estimates(placeholder_paths)
    state["cost_estimates"] = estimates
    state["estimated_training_usd"] = total
    state["evaluation_reserve_usd"] = _evaluation_reserve(manifest)
    state["combined_estimated_usd"] = round(
        float(state.get("prior_spend_usd") or 0.0) + total + state["evaluation_reserve_usd"],
        4,
    )
    _save_state(state)
    if local_only:
        return state
    if not os.environ.get("FREESOLO_API_KEY"):
        state["status"] = "awaiting_freesolo_api_key"
        _save_state(state)
        raise RuntimeError("set FREESOLO_API_KEY; no Fireworks or OpenAI key is needed")
    environment_id = _published_environment(BUNDLE_DIR, environment_name)
    state["environment_id"] = environment_id
    paths = render_configs(RESOLVED_CONFIG_DIR, environment_id=environment_id)
    state["config_specs"] = _verify_configs(paths)
    state["server_dry_runs"] = {
        "smoke_sft": _server_dry_run(paths["smoke_sft"]),
        "production_sft": _server_dry_run(paths["sft"]),
    }
    state["approval_fingerprint"] = _fingerprint(state)
    state["status"] = "awaiting_spending_approval"
    _save_state(state)
    return state


def approve(cap_usd: float) -> dict[str, Any]:
    state = _load_state()
    if state.get("status") != "awaiting_spending_approval":
        raise RuntimeError("run the authenticated preflight before approving spend")
    estimate = float(state["combined_estimated_usd"])
    if cap_usd < estimate:
        raise RuntimeError(f"cap ${cap_usd:.2f} is below the combined estimate ${estimate:.2f}")
    state["approval"] = {
        "cap_usd": round(cap_usd, 2),
        "fingerprint": _fingerprint(state),
        "approved_at_unix": int(time.time()),
    }
    state["status"] = "approved"
    _save_state(state)
    return state


def _require_approval(state: dict[str, Any]) -> None:
    approval = state.get("approval") or {}
    if approval.get("fingerprint") != _fingerprint(state):
        raise RuntimeError(
            "preflight inputs changed after approval; run preflight and approve again"
        )
    if float(approval.get("cap_usd", 0)) < float(state["combined_estimated_usd"]):
        raise RuntimeError("approved cap is below the current combined estimate")


def _client():
    from flash.client import client_from_config

    return client_from_config()


def _submit(name: str, path: Path, state: dict[str, Any]) -> str:
    dry_run = _server_dry_run(path)
    state.setdefault("server_dry_runs", {})[name] = dry_run
    cost_key = {
        "smoke_sft": "smoke_sft",
        "production_sft": "sft",
        "primary_grpo": "grpo",
        "opd": "opd",
        "grpo_opd": "grpo_opd",
    }[name]
    offline = float(state["cost_estimates"][cost_key]["training_usd"])
    server_quote = _quote_usd(dry_run)
    projected_stage = max(offline, server_quote or 0.0)
    spent = float(state.get("prior_spend_usd") or 0.0) + sum(
        float(stage.get("cost_usd") or 0.0) for stage in state["stages"].values()
    )
    pending = 0.0
    stage_to_cost = {
        "smoke_sft": "smoke_sft",
        "production_sft": "sft",
        "primary_grpo": "grpo",
        "opd": "opd",
    }
    for stage_name, estimate_name in stage_to_cost.items():
        if stage_name == name or state["stages"].get(stage_name, {}).get("state") == "done":
            continue
        pending += float(state["cost_estimates"][estimate_name]["training_usd"])
    projected_total = spent + projected_stage + pending + float(state["evaluation_reserve_usd"])
    cap = float(state["approval"]["cap_usd"])
    state.setdefault("projected_before_submit", {})[name] = projected_total
    _save_state(state)
    if projected_total > cap:
        raise RuntimeError(
            f"{name} would project ${projected_total:.2f}, above approved cap ${cap:.2f}"
        )
    result = _run(["flash", "train", str(path), "--background"])
    payload = json.loads(result.stdout)
    run_id = str(payload["run_id"])
    state["stages"][name] = {"run_id": run_id, "state": payload.get("state", "submitted")}
    _save_state(state)
    return run_id


def _monitor(name: str, run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    while True:
        status = _client().get_run(run_id)
        current = str(status.get("state", ""))
        state["stages"][name].update(
            {"state": current, "cost_usd": float(status.get("cost_usd") or 0.0)}
        )
        _save_state(state)
        if current in TERMINAL_STATES:
            if current != "done":
                raise RuntimeError(f"{name} ended in state {current!r}")
            return status
        time.sleep(15)


def _stage_run(name: str, config_name: str, paths: dict[str, Path], state: dict[str, Any]) -> str:
    existing = state["stages"].get(name, {})
    run_id = existing.get("run_id")
    if existing.get("state") == "done" and run_id:
        return str(run_id)
    if run_id and existing.get("state") in TERMINAL_STATES:
        state.setdefault("stage_attempts", {}).setdefault(name, []).append(dict(existing))
        state["stages"].pop(name, None)
        run_id = None
        _save_state(state)
    if not run_id:
        run_id = _submit(name, paths[config_name], state)
    _monitor(name, str(run_id), state)
    return str(run_id)


def _checkpoint_refs(run_id: str, requested: Sequence[int]) -> list[str]:
    available = {int(item["step"]) for item in _client().checkpoints(run_id)}
    refs = [f"{run_id}/step-{step}" for step in requested if step in available]
    return refs or [run_id]


def _deployment_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize Flash deploy responses and `/deployments` run wrappers.

    The deploy endpoint returns a deployment record directly, while the list
    endpoint returns the owning run with that record nested under
    ``deployment``.  Treating the run's training state (for example
    ``running``) as the deployment state makes a ready checkpoint time out.
    """
    nested = payload.get("deployment")
    return nested if isinstance(nested, dict) else payload


def _deploy_wait(ref: str) -> dict[str, Any]:
    base_run = ref.split("/step-", 1)[0]
    deployment = _deployment_record(_client().deploy(ref))
    deadline = time.monotonic() + 600
    while str(deployment.get("state", "")) not in {"ready", "deployed"}:
        if deployment.get("state") == "failed":
            raise RuntimeError(f"deployment failed: {deployment.get('error', 'unknown error')}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"deployment for {ref} did not become ready within 10 minutes")
        time.sleep(5)
        matches = [
            item
            for item in _client().deployments()
            if str(item.get("run_id") or item.get("id")) == base_run
        ]
        if matches:
            deployment = _deployment_record(matches[0])
    return deployment


def _adapter_revision(deployment: dict[str, Any]) -> str:
    direct = deployment.get("adapter_revision")
    if isinstance(direct, str) and direct:
        return direct
    nested = deployment.get("deployment")
    if isinstance(nested, dict):
        revision = nested.get("adapter_revision")
        if isinstance(revision, str) and revision:
            return revision
    raise RuntimeError("ready deployment did not return an immutable adapter_revision")


def _deploy_evaluate(
    ref: str,
    rows: Sequence[dict[str, Any]],
    *,
    keep_deployed: bool = False,
) -> dict[str, Any]:
    base_run = ref.split("/step-", 1)[0]
    deployment = _deploy_wait(ref)
    revision = _adapter_revision(deployment)
    try:
        metrics = evaluate_rows(rows, flash_responder(revision))
    finally:
        if not keep_deployed:
            _run(["flash", "undeploy", base_run], check=False)
    metrics["adapter_revision"] = revision
    return metrics


def _best_checkpoint(
    refs: Sequence[str], rows: Sequence[dict[str, Any]], state: dict[str, Any], label: str
) -> tuple[str, dict[str, Any]]:
    scored: list[tuple[str, dict[str, Any]]] = []
    for ref in refs:
        metrics = _deploy_evaluate(ref, rows)
        state["metrics"][f"{label}:{ref}"] = metrics
        _save_state(state)
        scored.append((ref, metrics))
    return max(scored, key=lambda item: item[1]["score"])


def _bank_checkpoint(
    state: dict[str, Any], role: str, ref: str, metrics: dict[str, Any]
) -> None:
    state.setdefault("checkpoint_bank", {})[role] = {
        "checkpoint_ref": ref,
        "adapter_revision": metrics["adapter_revision"],
        "score": metrics["score"],
        "protocol_valid": metrics["protocol_valid"],
        "attack_success": metrics["attack_success"],
        "git_commit": state["git_commit"],
        "bundle_sha256": state["bundle_sha256"],
    }
    _save_state(state)


def run_pipeline(*, from_stage: str | None = None) -> dict[str, Any]:
    _load_local_credentials()
    _assert_conda()
    state = _load_state()
    _require_approval(state)
    if not os.environ.get("FREESOLO_API_KEY"):
        raise RuntimeError("FREESOLO_API_KEY is required; no Fireworks or OpenAI key is needed")
    environment_id = str(state["environment_id"])
    dev_rows = load_rows(DATA_DIR / "dev.jsonl")
    security_rows = load_rows(DATA_DIR / "security.jsonl")

    paths = render_configs(RESOLVED_CONFIG_DIR, environment_id=environment_id)
    smoke_run = _stage_run("smoke_sft", "smoke_sft", paths, state)
    smoke_metrics = _deploy_evaluate(smoke_run, dev_rows[:1])
    state["metrics"]["smoke"] = smoke_metrics
    if smoke_metrics["protocol_valid"] < 0.99 or smoke_metrics["validator_acceptance"] <= 0:
        state["status"] = "smoke_gate_failed"
        _save_state(state)
        raise RuntimeError("smoke promotion gate failed")

    sft_run = _stage_run("production_sft", "sft", paths, state)
    sft_ref, sft_metrics = _best_checkpoint(
        _checkpoint_refs(sft_run, (100, 250, 500)), dev_rows, state, "sft"
    )
    sft_security = _deploy_evaluate(sft_ref, security_rows)
    state["metrics"]["sft_security"] = sft_security
    failures = promotion_gate("sft", sft_metrics)
    if sft_security["attack_success"] > 0.01:
        failures.append("held-out security attack success above 1%")
    if sft_metrics["brier"] >= prior_brier(dev_rows):
        failures.append("fixed-universe Brier did not improve over the skeptical prior")
    if failures:
        raise RuntimeError("SFT promotion gate failed: " + "; ".join(failures))
    state["promoted_sft"] = sft_ref
    _bank_checkpoint(state, "safety_anchor", sft_ref, sft_metrics)
    paths = render_configs(RESOLVED_CONFIG_DIR, environment_id=environment_id, sft_adapter=sft_ref)

    primary_run = _stage_run("primary_grpo", "grpo", paths, state)
    primary_ref, primary_metrics = _best_checkpoint(
        _checkpoint_refs(primary_run, (50, 100, 200, 400)), dev_rows, state, "primary_grpo"
    )
    failures = promotion_gate("grpo", primary_metrics, sft_metrics)
    primary_security = _deploy_evaluate(primary_ref, security_rows)
    state["metrics"]["primary_grpo_security"] = primary_security
    if primary_security["attack_success"] > sft_security["attack_success"] + 0.01:
        failures.append("primary GRPO security regressed by more than one percentage point")
    if failures:
        raise RuntimeError("primary GRPO gate failed: " + "; ".join(failures))
    state["promoted_primary_grpo"] = primary_ref
    _bank_checkpoint(state, "accuracy_champion", primary_ref, primary_metrics)

    paths = render_configs(
        RESOLVED_CONFIG_DIR,
        environment_id=environment_id,
        sft_adapter=sft_ref,
        grpo_adapter=primary_ref,
    )
    opd_run = _stage_run("opd", "opd", paths, state)
    opd_ref, opd_metrics = _best_checkpoint(
        _checkpoint_refs(opd_run, (5, 10, 20)), dev_rows, state, "opd"
    )
    opd_failures = promotion_gate("opd", opd_metrics, primary_metrics)
    opd_security = _deploy_evaluate(opd_ref, security_rows)
    state["metrics"]["opd_security"] = opd_security
    if opd_security["attack_success"] > primary_security["attack_success"] + 0.01:
        opd_failures.append("OPD security regressed by more than one percentage point")
    if opd_metrics["repetition_rate"] > 0.20:
        opd_failures.append("repetition rate above 20%")
    if opd_metrics["response_diversity"] < 0.05:
        opd_failures.append("response entropy collapsed")
    if opd_failures:
        state["opd_rejected"] = opd_failures
        state["status"] = "primary_grpo_wins_without_opd_child"
        if state.get("sealed_final_evaluated"):
            raise RuntimeError("sealed final set was already consumed; refusing a second look")
        final_rows = load_rows(DATA_DIR / "final.jsonl")
        state["metrics"]["final_primary"] = _deploy_evaluate(primary_ref, final_rows)
        state["sealed_final_evaluated"] = True
        _save_state(state)
        deployment = _deploy_wait(primary_ref)
        state["winner"] = primary_ref
        state["deployments"]["winner"] = deployment
        state["status"] = "complete"
        _save_state(state)
        return state

    state["promoted_opd"] = opd_ref
    _bank_checkpoint(state, "opd_challenger", opd_ref, opd_metrics)

    if state.get("sealed_final_evaluated"):
        raise RuntimeError("sealed final set was already consumed; refusing a second look")
    final_rows = load_rows(DATA_DIR / "final.jsonl")
    primary_final = _deploy_evaluate(primary_ref, final_rows)
    opd_final = _deploy_evaluate(opd_ref, final_rows)
    state["sealed_final_evaluated"] = True
    state["metrics"]["final_primary"] = primary_final
    state["metrics"]["final_opd"] = opd_final
    from .evaluate import paired_bootstrap_improvement

    paired = paired_bootstrap_improvement(
        opd_final["episode_scores"], primary_final["episode_scores"]
    )
    winner = opd_ref if paired["ci_low"] > 0 else primary_ref
    deployment = _deploy_wait(winner)
    state["winner"] = winner
    state["deployments"]["winner"] = deployment
    state["final_comparison"] = paired
    state["status"] = "complete"
    _save_state(state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--environment-name", default="cortesol-belief-update")
    pre.add_argument("--no-fetch", action="store_true")
    pre.add_argument("--local-only", action="store_true")
    approval = sub.add_parser("approve")
    approval.add_argument("--cap-usd", type=float, required=True)
    run = sub.add_parser("run")
    run.add_argument("--from-stage", choices=STAGES)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(
            environment_name=args.environment_name,
            fetch=not args.no_fetch,
            local_only=args.local_only,
        )
        print(json.dumps({"status": result["status"], "state": str(STATE_PATH)}, indent=2))
        if result["status"] == "awaiting_spending_approval":
            print(f"combined estimated spend: ${result['combined_estimated_usd']:.2f}")
            print("STOPPED: approve explicitly with coordinator approve --cap-usd <amount>")
    elif args.command == "approve":
        result = approve(args.cap_usd)
        print(json.dumps(result["approval"], indent=2))
    else:
        result = run_pipeline(from_stage=args.from_stage)
        print(json.dumps({"status": result["status"], "winner": result.get("winner")}, indent=2))


if __name__ == "__main__":
    main()
