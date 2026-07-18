from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
import tomllib

import pytest
from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentEpisode

from cortesol.core.ops import OP_NAMES, ProposedOps, ops_json_schema
from cortesol.train.bundle import build_bundle
from cortesol.train.config_artifacts import render_configs
from cortesol.train.coordinator import (
    _best_checkpoint,
    _deployment_record,
    _evaluation_reserve,
    _retry_flash,
)
from cortesol.train.datasets import (
    build_all,
    build_sft_rows,
    canonical_ops,
    episode_events_from_metadata,
    episode_row,
)
from cortesol.train.environment import BeliefUpdateEnv, _replay
from cortesol.train.evaluate import evaluate_rows, flash_responder
from cortesol.train.teacher_filter import (
    build_teacher_filtered_dataset,
    build_teacher_seed_dataset,
    rejection_sample_rows,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    path = tmp_path_factory.mktemp("training-data")
    manifest = build_all(path)
    return path, manifest


def _example(row):
    return TaskExample(
        record=row,
        input=row["input"],
        output=row["output"],
        metadata=row["metadata"],
    )


def test_deployment_state_uses_nested_record_not_parent_training_state():
    nested = {
        "run_id": "flash-run",
        "state": "running",
        "deployment": {
            "run_id": "flash-run",
            "state": "ready",
            "adapter_revision": "flash-run@step-100." + "a" * 40,
        },
    }
    assert _deployment_record(nested)["state"] == "ready"
    assert _deployment_record(nested)["adapter_revision"].startswith("flash-run@step-100")


def test_flash_retry_recovers_only_transient_connectivity(monkeypatch):
    monkeypatch.setattr("cortesol.train.coordinator.time.sleep", lambda _: None)
    attempts = iter(
        [
            RuntimeError("cannot reach the Flash service: temporary DNS failure"),
            RuntimeError("nodename nor servname provided"),
            {"state": "done"},
        ]
    )

    def flaky():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    assert _retry_flash(flaky, label="test") == {"state": "done"}
    with pytest.raises(RuntimeError, match="not authorized"):
        _retry_flash(lambda: (_ for _ in ()).throw(RuntimeError("not authorized")), label="test")


def test_checkpoint_selection_screens_all_and_full_gates_only_winner(monkeypatch):
    calls: list[tuple[str, int]] = []

    def evaluate(ref, rows):
        calls.append((ref, len(rows)))
        score = {"run/step-100": 0.2, "run/step-250": 0.9, "run/step-500": 0.8}[ref]
        return {
            "episodes": len(rows),
            "score": score,
            "exact_operations": score,
            "protocol_valid": 1.0,
            "attack_success": 0.0,
            "adapter_revision": ref + "@immutable",
        }

    monkeypatch.setattr("cortesol.train.coordinator._deploy_evaluate", evaluate)
    monkeypatch.setattr("cortesol.train.coordinator._save_state", lambda _: None)
    state = {"metrics": {}}
    refs = ["run/step-100", "run/step-250", "run/step-500"]
    rows = [{"row": index} for index in range(128)]

    winner, metrics = _best_checkpoint(refs, rows, state, "sft")

    assert winner == "run/step-250"
    assert metrics["episodes"] == 128
    assert calls == [
        ("run/step-100", 4),
        ("run/step-250", 4),
        ("run/step-500", 4),
        ("run/step-250", 128),
    ]
    assert state["checkpoint_shortlists"]["sft"]["winner"] == winner


def test_checkpoint_selection_reuses_completed_screen_and_full_metrics(monkeypatch):
    ref = "run/step-250"
    screen = {
        "episodes": 4,
        "score": 0.9,
        "exact_operations": 1.0,
        "protocol_valid": 1.0,
        "attack_success": 0.0,
    }
    full = {**screen, "episodes": 128}
    state = {
        "metrics": {
            f"sft_screen:{ref}": screen,
            f"sft:{ref}": full,
        }
    }
    monkeypatch.setattr(
        "cortesol.train.coordinator._deploy_evaluate",
        lambda *_: pytest.fail("completed metrics should be reused"),
    )
    monkeypatch.setattr("cortesol.train.coordinator._save_state", lambda _: None)

    winner, metrics = _best_checkpoint([ref], [{}] * 128, state, "sft")

    assert winner == ref
    assert metrics is full


def test_dataset_profile_is_complete_deterministic_and_sealed(generated, tmp_path):
    path, manifest = generated
    assert {name: item["rows"] for name, item in manifest["files"].items()} == {
        "sft_smoke": 64,
        "sft_train": 2800,
        "rl_train": 1024,
        "dev": 128,
        "final": 256,
        "security": 5,
    }
    second = build_all(tmp_path)
    assert {name: item["sha256"] for name, item in manifest["files"].items()} == {
        name: item["sha256"] for name, item in second["files"].items()
    }
    assert manifest["schema_hashes"]
    assert manifest["split_lineage"]
    assert all(item["input_tokens"]["max"] <= 2048 for item in manifest["files"].values())
    for split in manifest["files"]:
        for row in (path / f"{split}.jsonl").read_text().splitlines():
            payload = json.loads(row)
            assert set(payload) == {"input", "output", "metadata"}
            assert "sim_meta" not in payload["input"]
            assert "red_flags=" not in payload["input"]

    reserve = _evaluation_reserve(manifest)
    assert 0 < reserve < 27.2476


def test_sft_covers_every_class_and_operation(generated):
    path, _ = generated
    rows = [json.loads(line) for line in (path / "sft_train.jsonl").read_text().splitlines()]
    classes = {row["metadata"]["event_class"] for row in rows}
    names = {op["op"] for row in rows for op in json.loads(row["output"])["ops"]}
    assert len(classes) == 7
    assert names == set(OP_NAMES)


def test_sft_namespaces_are_diverse_without_cross_split_leakage(generated):
    path, _ = generated
    smoke = [json.loads(line) for line in (path / "sft_smoke.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (path / "sft_train.jsonl").read_text().splitlines()]
    assert len({row["metadata"]["entity_family"] for row in smoke}) == 8
    assert len({row["metadata"]["source_family"] for row in smoke}) == 7
    assert len({row["metadata"]["template_family"] for row in smoke}) == 5
    assert len({row["metadata"]["entity_family"] for row in train}) >= 16
    assert len({row["metadata"]["source_family"] for row in train}) >= 9
    assert len({row["metadata"]["template_family"] for row in train}) >= 13
    assert all(not row["metadata"]["entity_family"].startswith("DV") for row in smoke + train)


def test_sft_start_episode_uses_each_rows_exact_stateful_input():
    first, later = build_sft_rows()[:2]
    assert first["metadata"]["seed"] == later["metadata"]["seed"]
    assert first["input"] != later["input"]

    env = BeliefUpdateEnv()
    first_messages = env.start_episode(_example(first), "contract")
    later_messages = env.start_episode(_example(later), "contract")

    assert first_messages[-1]["content"] == first["input"]
    assert later_messages[-1]["content"] == later["input"]
    assert first_messages[-1]["content"] != later_messages[-1]["content"]


def test_teacher_rejection_sampling_keeps_only_gold_with_short_rationale(generated, tmp_path):
    path, _ = generated
    source = [json.loads(line) for line in (path / "sft_train.jsonl").read_text().splitlines()[:3]]
    gold_by_input = {row["input"]: json.loads(row["output"]) for row in source}

    def teacher(input_text, attempt):
        payload = dict(gold_by_input[input_text])
        if attempt == 0:
            return '{"think":"too short","ops":[]}'
        payload["think"] = "The measured context supports this bounded operation."
        return json.dumps(payload)

    accepted, report = rejection_sample_rows(
        source,
        generate=teacher,
        teacher="run@immutable-revision",
        cache_path=tmp_path / "teacher-cache.jsonl",
        k=2,
        workers=2,
    )
    assert len(accepted) == len(source)
    assert report["candidate_verdicts"] == {
        "accepted": 3,
        "gold_mismatch": 3,
    }
    for original, filtered in zip(source, accepted, strict=True):
        assert canonical_ops(ProposedOps.model_validate_json(filtered["output"])) == canonical_ops(
            ProposedOps.model_validate_json(original["output"])
        )
        assert filtered["metadata"]["supervision"] == "teacher_rejection_sampling"
        assert filtered["metadata"]["teacher_chosen_attempt"] == 1
        assert filtered["metadata"]["teacher_response_sha256"]


def test_teacher_rejection_sampling_cache_prevents_repeat_calls(generated, tmp_path):
    path, _ = generated
    row = json.loads((path / "sft_train.jsonl").read_text().splitlines()[0])
    payload = json.loads(row["output"])
    payload["think"] = "This is the correct bounded evidence operation."
    calls = 0

    def teacher(_input_text, _attempt):
        nonlocal calls
        calls += 1
        return json.dumps(payload)

    kwargs = {
        "generate": teacher,
        "teacher": "run@immutable-revision",
        "cache_path": tmp_path / "teacher-cache.jsonl",
        "k": 2,
        "workers": 1,
    }
    first, _ = rejection_sample_rows([row], **kwargs)
    second, _ = rejection_sample_rows([row], **kwargs)
    assert first == second
    assert calls == 2


def test_teacher_filtered_dataset_preserves_sealed_splits_and_records_lineage(generated, tmp_path):
    source, source_manifest = generated
    output = tmp_path / "teacher-data"
    gold = {
        row["input"]: json.loads(row["output"])
        for split in ("sft_smoke", "sft_train")
        for row in [
            json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines()
        ]
    }

    def teacher(input_text, _attempt):
        payload = dict(gold[input_text])
        payload["think"] = "The context warrants exactly this bounded ledger operation."
        return json.dumps(payload)

    manifest = build_teacher_filtered_dataset(
        source,
        output,
        generate=teacher,
        teacher="run@immutable-revision",
        cache_dir=tmp_path / "cache",
        k=1,
        workers=4,
    )
    assert manifest["supervision"] == "teacher_rejection_sampling"
    assert manifest["teacher_filter"]["reports"]["sft_train"]["accepted_rows"] == 2800
    assert manifest["files"]["sft_train"]["rows"] == 2800
    assert manifest["files"]["final"] == source_manifest["files"]["final"]
    teacher_row = json.loads((output / "sft_train.jsonl").read_text().splitlines()[0])
    assert teacher_row["metadata"]["teacher_adapter_revision"] == "run@immutable-revision"
    assert ProposedOps.model_validate_json(teacher_row["output"]).think


def test_teacher_seed_dataset_adds_bounded_rationales_without_changing_gold(generated, tmp_path):
    source, _ = generated
    output = tmp_path / "teacher-seed"
    manifest = build_teacher_seed_dataset(source, output)
    source_rows = (source / "sft_train.jsonl").read_text().splitlines()
    seeded_rows = (output / "sft_train.jsonl").read_text().splitlines()
    assert manifest["supervision"] == "simulator_gold_teacher_seed"
    assert len(source_rows) == len(seeded_rows) == 2800
    for original_text, seeded_text in zip(source_rows[:20], seeded_rows[:20], strict=True):
        original = ProposedOps.model_validate_json(json.loads(original_text)["output"])
        seeded = ProposedOps.model_validate_json(json.loads(seeded_text)["output"])
        assert canonical_ops(original) == canonical_ops(seeded)
        assert 3 <= len(seeded.think.split()) <= 32


def test_bundle_excludes_all_held_out_data(generated, tmp_path):
    path, _ = generated
    bundle = tmp_path / "bundle"
    first = build_bundle(bundle, data_dir=path)
    names = {item.name for item in bundle.rglob("*") if item.is_file()}
    assert not {"dev.jsonl", "final.jsonl", "security.jsonl"} & names
    assert {"sft_smoke.jsonl", "sft_train.jsonl", "rl_train.jsonl"} <= names
    second = build_bundle(bundle, data_dir=path)
    assert first["sha256"] == second["sha256"]
    from flash.cli.envpush import _copy_env_sidecars, _with_syspath_bootstrap

    published = tmp_path / "published"
    published.mkdir()
    entrypoint = bundle / "environment.py"
    (published / "environment.py").write_text(_with_syspath_bootstrap(entrypoint.read_text()))
    _copy_env_sidecars(bundle, published, entrypoint=entrypoint)
    assert (published / "cortesol_runtime_bundle.py").is_file()
    assert not (published / "cortesol").exists()
    probe = """
import importlib.util
from pathlib import Path
import sys
entry = Path(sys.argv[1]).resolve()
sys.path = [item for item in sys.path if 'Cortesol' not in item]
spec = importlib.util.spec_from_file_location('published_environment', entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import cortesol
assert cortesol.__file__ == '<cortesol-runtime:cortesol>'
assert module.BeliefUpdateEnv.__module__ == 'cortesol.train.environment'
env = module.load_environment(dataset_path=str(entry.parent / 'dataset' / 'sft_smoke.jsonl'))
assert len(env.dataset) == 64
"""
    subprocess.run(
        [sys.executable, "-c", probe, str(published / "environment.py")],
        check=True,
        cwd=tmp_path,
    )


def test_every_generated_toml_parses_in_flash_1_0_and_uses_exact_schema(tmp_path):
    from flash.schema import spec_from_file

    paths = render_configs(
        tmp_path,
        environment_id="owner/cortesol",
        sft_adapter="sft-run/step-500",
        grpo_adapter="grpo-run/step-400",
        opd_adapter="opd-run/step-20",
    )
    schema = json.loads((tmp_path / "ops.schema.json").read_text())
    assert schema == ops_json_schema()
    for name, path in paths.items():
        with path.open("rb") as handle:
            assert tomllib.load(handle)
        spec = spec_from_file(str(path))
        assert spec.thinking is False
        if name == "smoke_sft":
            assert spec.model == "Qwen/Qwen3.5-4B"
            assert spec.train.max_steps == 32
            assert spec.train.save_at_steps == (32,)
        if name in {"grpo", "opd", "grpo_opd"}:
            assert json.loads(spec.train.structured_outputs)["json"] == schema
            assert "lora_rank" not in spec.to_dict()["train"]
            expected_context = 12_288 if name == "opd" else 8_192
            assert spec.train.max_context_tokens == expected_context
            if name == "opd":
                assert spec.train.group_size == 1
                assert spec.train.init_from_adapter == "grpo-run/step-400"


def test_gold_episode_reconstructs_and_scores_perfect_actions():
    row = episode_row(
        10_000,
        split="rl_train",
        entity_family="RL",
        source_family="rl_source",
        template_family="rl_template_v1",
    )
    example = _example(row)
    env = BeliefUpdateEnv()
    messages = list(env.start_episode(example, "contract"))
    for event in episode_events_from_metadata(row["metadata"]):
        response = canonical_ops(ProposedOps(ops=event.sim_meta.gold_ops))
        messages.append({"role": "assistant", "content": response})
        step = env.step_episode(example, messages, response)
        messages.extend(step.messages)
    reward = env.score_episode(
        example,
        EnvironmentEpisode(messages=tuple(messages), response_text=response),
    )
    metrics = {item.name: item.score for item in reward.metrics}
    assert step.done
    assert metrics["exact_operations"] == 1.0
    assert metrics["parse_validity"] == 1.0
    assert metrics["attack_success"] == 0.0


def test_frozen_evaluator_matches_stateless_production_extractor_calls():
    row = episode_row(
        10_001,
        split="dev",
        entity_family="DV",
        source_family="dev_source",
        template_family="dev_template_v1",
    )
    events = episode_events_from_metadata(row["metadata"])
    responses = iter(
        canonical_ops(ProposedOps(ops=event.sim_meta.gold_ops)) for event in events
    )
    observed: list[tuple[str, ...]] = []

    def responder(messages):
        observed.append(tuple(message["role"] for message in messages))
        return next(responses)

    metrics = evaluate_rows([row], responder)
    assert observed == [("system", "user")] * len(events)
    assert metrics["protocol_valid"] == 1.0
    assert metrics["exact_operations"] == 1.0


def test_frozen_evaluator_parallelizes_isolated_episodes():
    rows = [
        episode_row(
            seed,
            split="dev",
            entity_family="DV",
            source_family="dev_source",
            template_family="dev_template_v1",
        )
        for seed in (10_002, 10_003)
    ]
    gold = {
        event.id: canonical_ops(ProposedOps(ops=event.sim_meta.gold_ops))
        for row in rows
        for event in episode_events_from_metadata(row["metadata"])
    }
    lock = threading.Lock()
    active = max_active = 0

    def responder(messages):
        nonlocal active, max_active
        incoming = messages[-1]["content"]
        evidence_line = next(
            line for line in incoming.splitlines() if line.startswith("evidence_id=")
        )
        evidence_id = evidence_line.split(" ", 1)[0].split("=", 1)[1]
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return gold[evidence_id]

    metrics = evaluate_rows(rows, responder)

    assert metrics["episodes"] == 2
    assert metrics["exact_operations"] == 1.0
    assert max_active >= 2


def test_flash_responder_retries_direct_socket_timeout(monkeypatch):
    monkeypatch.setenv("FREESOLO_API_KEY", "test-key")
    monkeypatch.setattr("cortesol.train.evaluate.time.sleep", lambda _: None)
    attempts = 0

    def urlopen(_request, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 180
        if attempts == 1:
            raise TimeoutError("read operation timed out")
        return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"ops\\":[]}"}}]}')

    monkeypatch.setattr("cortesol.train.evaluate.urllib.request.urlopen", urlopen)
    respond = flash_responder("run@immutable")

    assert respond([{"role": "user", "content": "event"}]) == '{"ops":[]}'
    assert attempts == 2


def test_malformed_and_forged_injection_actions_never_mutate_and_rollouts_isolate():
    row = episode_row(
        12,
        split="security",
        entity_family="SC",
        source_family="security_source",
        template_family="security_v1",
    )
    events = episode_events_from_metadata(row["metadata"])
    malformed = _replay(row["metadata"], ["not json"] * len(events))
    assert not any(malformed.parse_valid)
    injection_index = next(
        index
        for index, event in enumerate(events)
        if event.sim_meta.event_class.value == "injection"
    )
    responses = ['{"ops":[]}'] * len(events)
    claim = events[0].sim_meta.gold_ops[0]["claim_id"]
    injection = events[injection_index]
    responses[injection_index] = json.dumps(
        {
            "ops": [
                {
                    "op": "APPLY_EVIDENCE",
                    "claim_id": claim,
                    "direction": "+",
                    "strength": "strong",
                    "evidence_id": injection.id,
                }
            ]
        }
    )
    forged = _replay(row["metadata"], responses)
    assert forged.unauthorized_mutations == 0
    first = _replay(row["metadata"], responses[:3])
    second = _replay(row["metadata"], responses[:3])
    assert [claim.model_dump() for claim in first.kb.claims.values()] == [
        claim.model_dump() for claim in second.kb.claims.values()
    ]


def test_empty_or_early_episode_gets_no_missing_action_credit():
    row = episode_row(
        44,
        split="rl_train",
        entity_family="RL",
        source_family="rl_source",
        template_family="rl_template_v1",
    )
    reward = BeliefUpdateEnv().score_episode(
        _example(row),
        EnvironmentEpisode(
            messages=({"role": "assistant", "content": '{"ops":[]}'},),
            response_text='{"ops":[]}',
        ),
    )
    metrics = {item.name: item.score for item in reward.metrics}
    assert metrics["exact_operations"] == 0.0
    assert metrics["parse_validity"] == pytest.approx(1 / 24)


@pytest.mark.parametrize(
    "attack",
    (
        "reject_all",
        "accept_all",
        "empty",
        "verbosity",
        "state_omission",
        "forged_evidence",
        "flooding",
        "set_confidence",
    ),
)
def test_reward_exploit_responses_cannot_claim_exact_action_credit(attack):
    row = episode_row(
        77,
        split="rl_train",
        entity_family="RL",
        source_family="rl_source",
        template_family="rl_template_v1",
    )
    events = episode_events_from_metadata(row["metadata"])
    claims = sorted(
        op["claim_id"] for event in events for op in event.sim_meta.gold_ops if "claim_id" in op
    )[:3]
    responses = []
    for event in events:
        if attack == "reject_all":
            payload = {"ops": [{"op": "REJECT", "evidence_id": event.id, "reason": "unverifiable"}]}
        elif attack in {"accept_all", "flooding"}:
            count = 3 if attack == "accept_all" else 6
            payload = {
                "ops": [
                    {
                        "op": "APPLY_EVIDENCE",
                        "claim_id": claims[index % len(claims)],
                        "direction": "+",
                        "strength": "strong",
                        "evidence_id": event.id,
                    }
                    for index in range(count)
                ]
            }
        elif attack == "forged_evidence":
            payload = {
                "ops": [
                    {
                        "op": "APPLY_EVIDENCE",
                        "claim_id": claims[0],
                        "direction": "+",
                        "strength": "strong",
                        "evidence_id": "forged",
                    }
                ]
            }
        elif attack == "set_confidence":
            payload = {"ops": [{"op": "SET_CONFIDENCE", "claim_id": claims[0], "value": 1.0}]}
        else:
            payload = {"ops": []}
        text = json.dumps(payload)
        responses.append(f"Here is the answer: {text}" if attack == "verbosity" else text)

    replay = _replay(row["metadata"], responses)
    if attack == "reject_all":
        assert sum(replay.exact) <= 4  # only genuinely unverifiable events can match
    else:
        assert not any(replay.exact)
    if attack in {"verbosity", "set_confidence"}:
        assert not any(replay.parse_valid)
    if attack == "flooding":
        assert replay.rejected > 0
    assert replay.unauthorized_mutations == 0
