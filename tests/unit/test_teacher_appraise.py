"""Unit tests for the frontier-teacher appraisal seam (`train/teacher_appraise`).

Deterministic and NETWORK-FREE: every test uses StubTeacher or a tiny in-test
appraiser. Coverage: the k-sample self-consistency + screen-agreement filter
(keeps consistent/sensible, drops inconsistent), the append-only cache (no repeat
calls), the clean no-key degrade, and the proof that a gold appraisal compiles via
the REAL deterministic judge to the intended ledger op — across the full range."""

from __future__ import annotations

import pytest

from cortesol.core.assessment import EvidenceAssessment
from cortesol.core.domains import get_active_domain, set_active_domain
from cortesol.train import teacher_appraise as ta
from cortesol.train.appraisal_dataset import (
    CorpusPaper,
    augment_case_report,
    augment_extraordinary,
    augment_injection,
    augment_overclaiming,
)
from cortesol.train.teacher_filter import JsonlCandidateCache

pytestmark = pytest.mark.unit

_ABSTRACT = (
    "In a randomized, double-blind, placebo-controlled trial (n=1961, p<0.001), the "
    "treatment produced a robust effect that was independently replicated."
)


@pytest.fixture(autouse=True)
def _peptides_active():
    original = get_active_domain()
    set_active_domain("peptides")
    yield
    set_active_domain(original)


def _asmt(evidence_id: str, **overrides) -> EvidenceAssessment:
    base = dict(
        schema_version="2.0",
        evidence_id=evidence_id,
        in_scope="in_scope",
        instruction_attack=False,
        document_type="primary_study",
        study_design="randomized_controlled",
        subject="alpha compound",
        object=None,
        claim_summary="alpha compound works",
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
    base.update(overrides)
    return EvidenceAssessment(**base)


class _Fixed:
    """Appraiser that always returns a given assessment (evidence_id is rebound)."""

    def __init__(self, assessment: EvidenceAssessment):
        self.assessment = assessment
        self.calls = 0

    def appraise(self, abstract, evidence_id, *, sample):
        self.calls += 1
        return self.assessment.model_copy(update={"evidence_id": evidence_id})


class _PerSample:
    """Appraiser returning a different assessment per sample index."""

    def __init__(self, assessments: list[EvidenceAssessment]):
        self.assessments = assessments
        self.calls = 0

    def appraise(self, abstract, evidence_id, *, sample):
        self.calls += 1
        a = self.assessments[sample % len(self.assessments)]
        return a.model_copy(update={"evidence_id": evidence_id})


def test_stub_teacher_is_consistent_and_kept():
    outcome = ta.label_one(ta.StubTeacher(), _ABSTRACT, "e1", teacher="stub", k=3)
    assert outcome.kept
    assert outcome.agreement == 1.0
    assert outcome.assessment is not None
    # a clean replicated RCT compiles to a strong positive apply
    assert ta.op_signature(outcome.assessment, _ABSTRACT) == ("APPLY_EVIDENCE", "+", "strong")


def test_inconsistent_samples_are_dropped():
    disagreeing = [
        _asmt("x", study_design="randomized_controlled"),  # -> strong
        _asmt(
            "x",
            study_design="case_report",
            document_type="case_report",
            sample_size=1,
            independent_replication=False,
            preregistered=False,
        ),  # -> weak
        _asmt("x", claim_direction="refutes"),  # -> negative direction
    ]
    outcome = ta.label_one(_PerSample(disagreeing), _ABSTRACT, "x", teacher="t", k=3)
    assert not outcome.kept
    assert outcome.reason == "inconsistent"
    assert outcome.assessment is None
    assert outcome.agreement < 0.5


def test_majority_agreement_keeps_modal_reading():
    strong = _asmt("y")
    weak = _asmt(
        "y",
        study_design="case_report",
        document_type="case_report",
        sample_size=1,
        independent_replication=False,
        preregistered=False,
    )
    # two strong, one weak -> agreement 2/3 kept, modal is the strong signature
    outcome = ta.label_one(_PerSample([strong, strong, weak]), _ABSTRACT, "y", teacher="t", k=3)
    assert outcome.kept
    assert outcome.agreement == pytest.approx(2 / 3)
    assert ta.op_signature(outcome.assessment, _ABSTRACT) == ("APPLY_EVIDENCE", "+", "strong")


def test_cache_prevents_repeat_calls(tmp_path):
    app = _Fixed(_asmt("e"))
    cache = JsonlCandidateCache(tmp_path / "cache.jsonl")
    first = ta.label_one(app, _ABSTRACT, "e", teacher="t", k=3, cache=cache)
    after_first = app.calls
    second = ta.label_one(app, _ABSTRACT, "e", teacher="t", k=3, cache=cache)
    assert after_first == 3
    assert app.calls == 3  # no new calls on the second pass — all served from cache
    assert first.assessment.model_dump() == second.assessment.model_dump()


def test_no_valid_sample_is_dropped():
    class _Boom:
        def appraise(self, abstract, evidence_id, *, sample):
            raise RuntimeError("provider down")

    outcome = ta.label_one(_Boom(), _ABSTRACT, "z", teacher="t", k=2)
    assert not outcome.kept
    assert outcome.reason == "no_valid_sample"


def test_frontier_teacher_requires_key_and_degrades_cleanly(monkeypatch):
    monkeypatch.setattr(ta, "_env", lambda *a, **k: None)
    assert ta.teacher_available() is False
    with pytest.raises(ta.TeacherKeyRequired):
        ta.FrontierTeacher()


def test_gold_appraisal_compiles_to_intended_op():
    paper = CorpusPaper(
        "arxiv:1", "physics", "Widget resonance study",
        "We measured widget resonance in n=200 samples (p=0.002).", "preprint",
    )
    inj = augment_injection(paper)
    assert ta.op_signature(inj.assessment, inj.text) == ("REJECT", "injection")

    case = augment_case_report(paper)
    assert ta.op_signature(case.assessment, case.text) == ("APPLY_EVIDENCE", "+", "weak")

    over = augment_overclaiming(paper)
    assert ta.op_signature(over.assessment, over.text) == ("APPLY_EVIDENCE", "+", "weak")

    extra = augment_extraordinary(paper)
    assert ta.op_signature(extra.assessment, extra.text) == ("APPLY_EVIDENCE", "+", "weak")


def test_label_batch_reports_keep_rate(tmp_path):
    items = [(_ABSTRACT, "a"), (_ABSTRACT, "b")]
    outcomes, report = ta.label_batch(
        ta.StubTeacher(), items, teacher="stub", k=3, cache_path=tmp_path / "c.jsonl"
    )
    assert report["items"] == 2
    assert report["kept"] == 2
    assert report["keep_rate"] == 1.0
    assert all(o.kept for o in outcomes)
