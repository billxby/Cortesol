from __future__ import annotations

import json
import tomllib

import pytest
from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentEpisode

from cortesol.core.ops import OP_NAMES, ProposedOps, ops_json_schema
from cortesol.train.bundle import build_bundle
from cortesol.train.config_artifacts import render_configs
from cortesol.train.datasets import (
    build_all,
    canonical_ops,
    episode_events_from_metadata,
    episode_row,
)
from cortesol.train.environment import BeliefUpdateEnv, _replay

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


def test_sft_covers_every_class_and_operation(generated):
    path, _ = generated
    rows = [json.loads(line) for line in (path / "sft_train.jsonl").read_text().splitlines()]
    classes = {row["metadata"]["event_class"] for row in rows}
    names = {op["op"] for row in rows for op in json.loads(row["output"])["ops"]}
    assert len(classes) == 7
    assert names == set(OP_NAMES)


def test_bundle_excludes_all_held_out_data(generated, tmp_path):
    path, _ = generated
    bundle = tmp_path / "bundle"
    first = build_bundle(bundle, data_dir=path)
    names = {item.name for item in bundle.rglob("*") if item.is_file()}
    assert not {"dev.jsonl", "final.jsonl", "security.jsonl"} & names
    assert {"sft_smoke.jsonl", "sft_train.jsonl", "rl_train.jsonl"} <= names
    second = build_bundle(bundle, data_dir=path)
    assert first["sha256"] == second["sha256"]


def test_every_generated_toml_parses_in_flash_1_0_and_uses_exact_schema(tmp_path):
    from flash.schema import spec_from_file

    paths = render_configs(
        tmp_path,
        environment_id="owner/cortesol",
        sft_adapter="sft-run/step-500",
        opd_adapter="opd-run/step-20",
    )
    schema = json.loads((tmp_path / "ops.schema.json").read_text())
    assert schema == ops_json_schema()
    for name, path in paths.items():
        with path.open("rb") as handle:
            assert tomllib.load(handle)
        spec = spec_from_file(str(path))
        assert spec.thinking is False
        if name in {"grpo", "opd", "grpo_opd"}:
            assert json.loads(spec.train.structured_outputs)["json"] == schema
            assert "lora_rank" not in spec.to_dict()["train"]


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
