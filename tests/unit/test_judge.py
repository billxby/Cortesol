"""Unit tests for the deterministic, SUBJECT-BLIND judge (`core/judge.py::judge`).

The judge compiles a domain-general critical-appraisal EvidenceAssessment into
ProposedOps by fixed rules — strength comes from methodology, never from topic.
Tests run offline against the contract types, and the last block proves the
generalization property: the SAME rules apply to peptide AND materials claims."""

from __future__ import annotations

import pytest

from cortesol.core.assessment import EvidenceAssessment
from cortesol.core.context import Context
from cortesol.core.domains import get_active_domain, set_active_domain
from cortesol.core.judge import judge
from cortesol.core.kb import KB
from cortesol.core.ops import AddClaim, ApplyEvidence, FlagOOD, Reject
from cortesol.core.schema import Claim, Evidence, RawEvent

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _peptides_active():
    """The active domain is process-global; pin it to peptides for these tests."""
    original = get_active_domain()
    set_active_domain("peptides")
    yield
    set_active_domain(original)


def _assessment(evidence_id: str = "E", **overrides) -> EvidenceAssessment:
    """A clean, strong, well-powered study (RCT + replicated + preregistered);
    override fields per test."""
    base = dict(
        schema_version="2.0",
        evidence_id=evidence_id,
        in_scope="in_scope",
        instruction_attack=False,
        document_type="primary_study",
        study_design="randomized_controlled",
        subject="P1",
        object="MC4R",
        claim_summary="P1 binds MC4R",
        claim_direction="supports",
        magnitude="Kd 12 nM",
        value=12.0,
        value_relation="approximate",
        units="nM",
        sample_size=3,
        replicate_count=3,
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


def _claim(cid: str, text: str, tags: list[str]) -> Claim:
    return Claim(id=cid, text=text, ontology_tags=tags)


def _ctx(claims: list[Claim], *, evidence_id: str = "E", raw_text: str = "P1 binds MC4R"):
    kb = KB()
    for claim in claims:
        kb.add_claim(claim)
    evidence = Evidence(id=evidence_id, source_id="S", raw_text=raw_text)
    event = RawEvent(id=evidence_id, t=0, source_id="S", raw_text=raw_text)
    return kb, Context(event=event, evidence=evidence, claims=claims, sources=[], edges=[])


def _peptide_claim() -> Claim:
    return _claim(
        "c_bind_P1_MC4R", "P1 binds MC4R", ["peptide:P1", "target:MC4R", "binding_affinity:Kd"]
    )


def test_instruction_attack_rejects_injection():
    kb, ctx = _ctx([_peptide_claim()])
    out = judge(kb, ctx, _assessment(instruction_attack=True))
    assert len(out.ops) == 1
    assert isinstance(out.ops[0], Reject) and out.ops[0].reason == "injection"


def test_out_of_scope_flags_ood():
    kb, ctx = _ctx([])
    out = judge(kb, ctx, _assessment(in_scope="out_of_scope"))
    assert len(out.ops) == 1 and isinstance(out.ops[0], FlagOOD)


def test_not_a_claim_flags_ood():
    kb, ctx = _ctx([])
    out = judge(kb, ctx, _assessment(claim_direction="not_a_claim"))
    assert isinstance(out.ops[0], FlagOOD)


def test_clean_strong_applies_evidence_to_matched_claim():
    kb, ctx = _ctx([_peptide_claim()])
    out = judge(kb, ctx, _assessment())
    op = out.ops[0]
    assert isinstance(op, ApplyEvidence)
    assert op.claim_id == "c_bind_P1_MC4R"
    assert op.direction == "+"
    assert op.strength == "strong"
    assert op.evidence_id == "E"


def test_no_match_mints_new_claim_in_active_ontology():
    kb, ctx = _ctx([])  # nothing to match against
    out = judge(kb, ctx, _assessment(subject="P2", object="GLP1R", claim_summary="P2 binds GLP1R"))
    op = out.ops[0]
    assert isinstance(op, AddClaim)
    assert op.text == "P2 binds GLP1R"
    assert op.ontology_tags[0] == "peptide:p2"  # active domain's primary entity type
    assert op.initial_evidence_id == "E"


def test_null_result_is_negative_direction():
    kb, ctx = _ctx([_peptide_claim()])
    out = judge(kb, ctx, _assessment(claim_direction="null_result"))
    assert isinstance(out.ops[0], ApplyEvidence) and out.ops[0].direction == "-"


# --- the generalization property: strength is driven by METHOD, not subject -----

def _materials_ctx():
    claim = _claim("c_mat_x", "material X is superconducting at 300 K", [])
    return _ctx([claim], raw_text="material X superconducts at 300 K")


def _materials(**overrides) -> EvidenceAssessment:
    base = dict(
        subject="material X",
        object="superconductivity",
        claim_summary="material X is superconducting at 300 K",
    )
    base.update(overrides)
    return _assessment(**base)


def test_strength_subject_blind_strong_for_replicated_rct():
    # A DIFFERENT field, same clean methodology → the same "strong" the peptide RCT got.
    kb, ctx = _materials_ctx()
    op = judge(kb, ctx, _materials()).ops[0]
    assert isinstance(op, ApplyEvidence) and op.claim_id == "c_mat_x"
    assert op.strength == "strong"


def test_strength_subject_blind_weak_for_underpowered_case_report():
    kb, ctx = _materials_ctx()
    op = judge(
        kb,
        ctx,
        _materials(
            study_design="case_report",
            document_type="case_report",
            sample_size=1,
            randomized=False,
            controlled=False,
            blinded=False,
            preregistered=False,
            independent_replication=False,
            p_value=None,
            confidence_interval_reported=False,
            effect_size_reported=False,
        ),
    ).ops[0]
    assert isinstance(op, ApplyEvidence) and op.claim_id == "c_mat_x"
    assert op.strength == "weak"  # a case report barely moves belief — in ANY field


def test_extraordinary_claim_without_strong_support_is_weak():
    kb, ctx = _materials_ctx()
    op = judge(
        kb,
        ctx,
        _materials(
            study_design="observational",
            document_type="preprint",
            independent_replication=False,
            preregistered=False,
            extraordinary_claim=True,
        ),
    ).ops[0]
    assert op.strength == "weak"  # extraordinary claims need extraordinary evidence
