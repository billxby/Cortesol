"""Unit tests for the deterministic judge (`core/judge.py::judge`).

The judge is the epistemic policy boundary: it compiles a neutral, model-filled
EvidenceAssessment into ProposedOps entirely by fixed rules. These tests exercise
the four canonical outcomes directly (no model, no network), building the KB /
Context / Evidence with main's own contract types."""

from __future__ import annotations

import pytest

from cortesol.core.assessment import EvidenceAssessment
from cortesol.core.context import Context
from cortesol.core.judge import judge
from cortesol.core.kb import KB
from cortesol.core.ops import AddClaim, ApplyEvidence, FlagOOD, ProposedOps, Reject
from cortesol.core.schema import Claim, Evidence, RawEvent

pytestmark = pytest.mark.unit


def _assessment(evidence_id: str = "E", **overrides) -> EvidenceAssessment:
    """A clean, well-powered binding assessment; override fields per test."""
    base = dict(
        schema_version="1.0",
        evidence_id=evidence_id,
        scope="peptide",
        instruction_attack=False,
        document_type="primary_research",
        study_type="binding",
        peptide="P1",
        target_or_indication="MC4R",
        property="binding_affinity",
        assay="SPR",
        endpoint="Kd",
        finding="observed",
        value=12.0,
        value_relation="approximately",
        units="nM",
        sample_size=3,
        replicate_count=3,
        p_value=0.001,
        randomized=None,
        blinded=None,
        controlled=True,
        preregistered=True,
        control_peptide=True,
        purity_pct=98.0,
    )
    base.update(overrides)
    return EvidenceAssessment(**base)


def _ctx(claims: list[Claim], *, evidence_id: str = "E", raw_text: str = "P1 binds MC4R"):
    kb = KB()
    for claim in claims:
        kb.add_claim(claim)
    evidence = Evidence(id=evidence_id, source_id="S", raw_text=raw_text)
    event = RawEvent(id=evidence_id, t=0, source_id="S", raw_text=raw_text)
    ctx = Context(event=event, evidence=evidence, claims=claims, sources=[], edges=[])
    return kb, ctx


def _binding_claim() -> Claim:
    return Claim(
        id="c_bind_P1_MC4R",
        text="P1 binds MC4R",
        ontology_tags=["peptide:P1", "target:MC4R", "binding_affinity:Kd"],
    )


def test_instruction_attack_compiles_to_single_injection_reject():
    kb, ctx = _ctx([_binding_claim()])
    out = judge(kb, ctx, _assessment(instruction_attack=True))
    assert isinstance(out, ProposedOps)
    assert len(out.ops) == 1
    op = out.ops[0]
    assert isinstance(op, Reject)
    assert op.reason == "injection"
    assert op.evidence_id == "E"


def test_non_peptide_scope_compiles_to_flag_ood():
    kb, ctx = _ctx([], raw_text="Aspirin inhibits COX-1 in platelets")
    out = judge(kb, ctx, _assessment(scope="non_peptide"))
    assert len(out.ops) == 1
    assert isinstance(out.ops[0], FlagOOD)


def test_clean_assessment_applies_evidence_to_matched_claim():
    kb, ctx = _ctx([_binding_claim()])
    out = judge(kb, ctx, _assessment())
    assert len(out.ops) == 1
    op = out.ops[0]
    assert isinstance(op, ApplyEvidence)
    assert op.claim_id == "c_bind_P1_MC4R"
    assert op.direction == "+"
    assert op.strength == "strong"
    assert op.evidence_id == "E"


def test_clean_assessment_without_match_adds_new_claim():
    kb, ctx = _ctx([])  # nothing in context to match against
    out = judge(
        kb,
        ctx,
        _assessment(peptide="P2", target_or_indication="GLP1R", property="binding_affinity"),
    )
    assert len(out.ops) == 1
    op = out.ops[0]
    assert isinstance(op, AddClaim)
    assert op.text == "P2 binds GLP1R"
    assert op.initial_evidence_id == "E"
    assert "peptide:P2" in op.ontology_tags
