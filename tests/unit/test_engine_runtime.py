from __future__ import annotations

import math

import pytest

from cortesol.core import config
from cortesol.core.engine import apply_evidence
from cortesol.core.kb import KB
from cortesol.core.ops import AddEdge, ApplyEvidence, InvalidateEdge, ProposedOps
from cortesol.core.propagate import discredit_source
from cortesol.core.schema import Claim, ClaimStatus, EdgeType, Evidence, Source
from cortesol.pipeline import commit_proposal, prepare_event
from cortesol.sim.events import emit_stream
from cortesol.train.datasets import initial_kb

pytestmark = pytest.mark.unit


def _kb(tier: str = "reputable") -> tuple[KB, Evidence]:
    kb = KB()
    kb.add_claim(Claim(id="P1", text="P1 binds T", ontology_tags=["peptide:P1", "binding:T"]))
    kb.add_source(Source.from_tier("S", tier))
    return kb, Evidence(id="E", source_id="S", correlation_group="lab|SPR|ds")


def test_updates_are_source_capped_and_bounded():
    kb, evidence = _kb("weak")
    before = kb.claims["P1"].ell
    delta = apply_evidence(
        kb,
        ApplyEvidence(claim_id="P1", direction="+", strength="strong", evidence_id="E"),
        evidence,
    )
    tau, phi = config.SOURCE_PRIORS["weak"]
    assert delta.after_ell - before <= math.log(tau / phi)
    assert abs(delta.after_ell - before) <= config.DELTA_MAX


def test_red_flags_and_correlated_echoes_saturate():
    clean_kb, clean = _kb()
    flagged_kb, flagged = _kb()
    flagged.red_flags = ["grim_fail", "predatory_venue", "low_purity"]
    op = ApplyEvidence(claim_id="P1", direction="+", strength="strong", evidence_id="E")
    clean_first = apply_evidence(clean_kb, op, clean)
    flagged_first = apply_evidence(flagged_kb, op, flagged)
    assert flagged_first.after_ell - flagged_first.before_ell < (
        clean_first.after_ell - clean_first.before_ell
    )
    echo = Evidence(id="E2", source_id="S", correlation_group="lab|SPR|ds")
    echo_delta = apply_evidence(
        clean_kb,
        ApplyEvidence(claim_id="P1", direction="+", strength="strong", evidence_id="E2"),
        echo,
    )
    assert (
        echo_delta.after_ell - echo_delta.before_ell
        < clean_first.after_ell - clean_first.before_ell
    )


def test_validator_requires_current_provenance_and_blocks_injection_mutation():
    event = emit_stream(0, 7)[-1]
    kb = initial_kb(0)
    ctx = prepare_event(kb, event)
    claim_id = sorted(kb.claims)[0]
    proposal = ProposedOps(
        ops=[
            ApplyEvidence(
                claim_id=claim_id,
                direction="+",
                strength="strong",
                evidence_id=event.id,
            )
        ]
    )
    before = kb.claims[claim_id].ell
    result = commit_proposal(kb, ctx, proposal)
    assert result.validation.all_rejected
    assert kb.claims[claim_id].ell == before
    assert not result.deltas


def test_conflict_quarantines_instead_of_fusing():
    kb, evidence = _kb()
    claim = kb.claims["P1"]
    claim.r = 20
    proposal = ProposedOps(
        ops=[ApplyEvidence(claim_id="P1", direction="-", strength="strong", evidence_id="E")]
    )
    from cortesol.core.validator import validate

    result = validate(kb, proposal, evidence)
    assert result.all_rejected
    assert claim.status is ClaimStatus.QUARANTINED


def test_edge_is_invalidated_not_deleted_and_discredit_reverses_support():
    kb, evidence = _kb()
    kb.add_claim(Claim(id="P2", text="P2 is stable", ontology_tags=["peptide:P2", "stability:x"]))
    ctx_event = emit_stream(2, 1)[0].model_copy(
        update={"id": "E", "source_id": "S", "sim_meta": None}
    )
    ctx = prepare_event(kb, ctx_event)
    add = ProposedOps(ops=[AddEdge(src="P1", dst="P2", type=EdgeType.SUPPORTS, weight=0.8)])
    commit_proposal(kb, ctx, add)
    edge_id = next(iter(kb.edges))
    commit_proposal(
        kb,
        ctx,
        ProposedOps(ops=[InvalidateEdge(edge_id=edge_id, evidence_id="E", reason="retracted")]),
    )
    assert edge_id in kb.edges and not kb.edges[edge_id].live

    apply_evidence(
        kb,
        ApplyEvidence(claim_id="P1", direction="+", strength="moderate", evidence_id="E"),
        evidence,
    )
    supported = kb.claims["P1"].ell
    discredit_source(kb, "S")
    assert kb.sources["S"].discredited
    assert kb.claims["P1"].ell < supported
