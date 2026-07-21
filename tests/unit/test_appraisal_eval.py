"""Unit tests for the held-out-FIELD appraisal eval (`eval/appraisal_eval`).

Deterministic and NETWORK-FREE: every student is an offline stand-in (StubTeacher,
the committed teacher labels, or a tiny in-test fake) and every metric is computed
through the REAL deterministic judge+engine. Coverage: op-signature agreement on the
held-out fields (perfect student -> 1.0; a direction-flipping student drops below),
attack-resistance (injected text moves zero belief even for a credulous student — the
ledger screen, not just the model), and the known-cases judge+engine scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortesol.core.assessment import EvidenceAssessment
from cortesol.core.domains import get_active_domain, set_active_domain
from cortesol.eval.appraisal_eval import (
    INJECTION_KIND,
    OfflineAppraiser,
    attack_resistance,
    build_student,
    evaluate,
    judge_and_commit,
    known_case_scores,
    op_signature_agreement,
)
from cortesol.train import teacher_appraise as ta
from cortesol.train.appraisal_dataset import (
    HELDOUT_FIELDS,
    CorpusPaper,
    appraisal_examples_from_labels,
    augment_injection,
)

pytestmark = pytest.mark.unit

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_REPLICATION = (
    "In a randomized, double-blind, placebo-controlled trial (n=1961, p<0.001) the "
    "treatment produced a robust effect that was independently replicated."
)


@pytest.fixture(autouse=True)
def _peptides_active():
    original = get_active_domain()
    set_active_domain("peptides")
    yield
    set_active_domain(original)


@pytest.fixture(scope="module")
def heldout():
    _, examples = appraisal_examples_from_labels(_DATA_DIR)
    return examples


class _GoldStudent:
    """Echoes the teacher gold assessment for every held-out example."""

    label = "gold"

    def __init__(self, examples):
        self._by_id = {e.evidence_id: e.assessment for e in examples}

    def appraise(self, text, evidence_id):
        return self._by_id[evidence_id].model_copy(update={"evidence_id": evidence_id})


class _FlipStudent(_GoldStudent):
    """The teacher gold with the claim direction flipped — forces op mismatches."""

    label = "flip"

    def appraise(self, text, evidence_id):
        gold = self._by_id[evidence_id]
        flip = {"supports": "refutes", "refutes": "supports"}.get(
            gold.claim_direction, gold.claim_direction
        )
        return gold.model_copy(update={"evidence_id": evidence_id, "claim_direction": flip})


class _CredulousStudent:
    """Always reports a clean replicated RCT and never flags an attack. The screen
    must still stop injected text — this is the defense-in-depth assertion."""

    label = "credulous"

    def appraise(self, text, evidence_id):
        return EvidenceAssessment(
            schema_version="2.0",
            evidence_id=evidence_id,
            in_scope="in_scope",
            instruction_attack=False,
            document_type="primary_study",
            study_design="randomized_controlled",
            subject="thing",
            object=None,
            claim_summary="thing works",
            claim_direction="supports",
            magnitude=None,
            value=None,
            value_relation="not_reported",
            units=None,
            sample_size=200,
            replicate_count=None,
            p_value=0.001,
            confidence_interval_reported=True,
            effect_size_reported=True,
            controlled=True,
            randomized=True,
            blinded=True,
            preregistered=True,
            independent_replication=True,
            extraordinary_claim=False,
            overclaiming=False,
        )


def test_op_signature_agreement_perfect_student_is_one(heldout):
    metric = op_signature_agreement(_GoldStudent(heldout), heldout)
    assert metric["n"] > 0
    assert metric["agreement"] == 1.0
    assert not metric["mismatches"]
    # per-field breakdown covers only held-out fields, each fully agreeing
    assert set(metric["by_field"]) <= set(HELDOUT_FIELDS)
    assert all(row["agreement"] == 1.0 for row in metric["by_field"].values())


def test_op_signature_agreement_drops_and_records_mismatches(heldout):
    perfect = op_signature_agreement(_GoldStudent(heldout), heldout)
    flipped = op_signature_agreement(_FlipStudent(heldout), heldout)
    assert flipped["agreement"] < perfect["agreement"] == 1.0
    assert flipped["mismatches"], "flipping the direction must produce recorded mismatches"
    for mismatch in flipped["mismatches"]:
        assert mismatch["field"] in HELDOUT_FIELDS
        assert mismatch["teacher"] != mismatch["student"]


def test_attack_resistance_holds_even_for_a_credulous_student(heldout):
    injections = [e for e in heldout if e.kind == INJECTION_KIND]
    assert injections, "held-out fields must contribute injection cases"

    stub = OfflineAppraiser(ta.StubTeacher(), label="stub")
    stub_metric = attack_resistance(stub, heldout)
    assert stub_metric["n"] == len(injections)
    assert stub_metric["attack_success"] == 0.0
    assert not stub_metric["failures"]

    # the deterministic screen rejects injected text even when the model misses it,
    # so belief moves by zero regardless of the student.
    credulous = attack_resistance(_CredulousStudent(), heldout)
    assert credulous["resisted"] == credulous["n"] == len(injections)
    assert credulous["attack_success"] == 0.0


def test_known_cases_scored_by_judge_and_engine():
    metric = known_case_scores(OfflineAppraiser(ta.StubTeacher(), label="stub"), str(_DATA_DIR))
    assert metric["n"] == metric["high"]["n"] + metric["low"]["n"]
    assert metric["high"]["n"] >= 1 and metric["low"]["n"] >= 1
    assert metric["accuracy"] == 1.0
    for detail in metric["details"]:
        assert detail["belief"] is not None  # engine-committed confidence
        assert detail["correct"] is True
        if detail["expected"] == "high":
            assert tuple(detail["signature"])[0] == "APPLY_EVIDENCE"


def test_judge_and_commit_moves_belief_for_replication_not_for_injection():
    strong = ta.stub_appraise(_REPLICATION, "strong")
    _, belief = judge_and_commit(strong, _REPLICATION)
    assert belief is not None and belief > 0.45  # rose above the skeptical prior (c0=0.45)

    paper = CorpusPaper("arxiv:z", "math", "Widget", "We measured x (n=50, p=0.01).", "preprint")
    injection = augment_injection(paper)
    result, _ = judge_and_commit(injection.assessment, injection.text)
    assert not result.deltas  # injected text commits zero belief


def test_evaluate_is_deterministic_and_offline():
    student = build_student(offline="stub")
    first = evaluate(student, data_dir=str(_DATA_DIR))
    second = evaluate(student, data_dir=str(_DATA_DIR))
    assert first == second
    assert first["held_out_fields"] == list(HELDOUT_FIELDS)
    assert first["op_signature_agreement"]["n"] > 0
    assert first["known_cases"]["accuracy"] == 1.0
