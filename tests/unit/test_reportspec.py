"""Offline tests for the grounded generative report (cortesol/ui/reportspec.py).

Exercises the deterministic core only — NO network, NO LLM. Verifies that:
  * render_report reads numbers straight off the ledger (never fabricates),
  * a missing claim id / tag renders as an explicit ungrounded component,
  * groundedness (the output-side immune system) catches a bogus claim id and an
    uncited prose_block,
  * default_report produces a fully-grounded spec from the KB.
"""

from __future__ import annotations

import pytest

from cortesol.core.kb import KB
from cortesol.core.mathx import logit, sigmoid
from cortesol.core.schema import Claim, ClaimStatus, Edge, EdgeType, TrajectoryPoint
from cortesol.ui.reportspec import (
    ClaimCard,
    ConfidenceMeter,
    ContradictionPanel,
    EvidenceTable,
    MetricComparison,
    ProseBlock,
    ProvenanceTrail,
    ReportSpec,
    StatTile,
    TrajectorySparkline,
    default_report,
    groundedness,
    render_report,
)


@pytest.fixture
def kb() -> KB:
    """A small belief graph: two well-supported claims joined by a live contradicts
    edge, each carrying a two-point trajectory with human-readable causes."""
    k = KB()
    c1 = Claim(
        id="c_bind_P1_MC4R",
        text="P1 binds MC4R",
        ontology_tags=["peptide:p1", "target:mc4r"],
        ell=logit(0.9),
        r=5,
        s=1,
        status=ClaimStatus.ACTIVE,
    )
    c1.trajectory = [
        TrajectoryPoint(t=0, ell=0.2, cause="seed from prior"),
        TrajectoryPoint(t=1, ell=logit(0.9), cause="APPLY_EVIDENCE from Nature [strong]"),
    ]
    c2 = Claim(
        id="c_efficacy_P1",
        text="P1 is efficacious in vivo",
        ontology_tags=["peptide:p1"],
        ell=logit(0.4),
        r=2,
        s=3,
        status=ClaimStatus.FLAGGED,
    )
    c2.trajectory = [
        TrajectoryPoint(t=2, ell=logit(0.4), cause="APPLY_EVIDENCE from bioRxiv [weak]")
    ]
    k.add_claim(c1)
    k.add_claim(c2)
    k.add_edge(
        Edge(
            id="e1",
            src="c_bind_P1_MC4R",
            dst="c_efficacy_P1",
            type=EdgeType.CONTRADICTS,
            weight=0.8,
        )
    )
    return k


# --------------------------------------------------------------------------
# render_report: numbers must equal the ledger's, verbatim.
# --------------------------------------------------------------------------


def test_render_claim_card_matches_ledger(kb: KB):
    spec = ReportSpec(title="t", components=[ClaimCard(claim_id="c_bind_P1_MC4R")])
    out = render_report(spec, kb)
    comp = out["components"][0]
    assert comp["grounded"] is True
    claim = kb.get_claim("c_bind_P1_MC4R")
    assert comp["claim"]["c"] == round(claim.c, 4)
    assert comp["claim"]["u"] == round(claim.u, 4)
    assert comp["claim"]["r"] == claim.r == 5
    assert comp["claim"]["s"] == claim.s == 1
    assert comp["claim"]["status"] == "active"
    assert comp["claim"]["text"] == "P1 binds MC4R"


def test_render_confidence_meter_and_stat_tiles(kb: KB):
    spec = ReportSpec(
        title="t",
        components=[
            ConfidenceMeter(claim_id="c_bind_P1_MC4R"),
            StatTile(claim_id="c_bind_P1_MC4R", stat="c"),
            StatTile(claim_id="c_bind_P1_MC4R", stat="u"),
            StatTile(claim_id="c_bind_P1_MC4R", stat="r"),
            StatTile(claim_id="c_bind_P1_MC4R", stat="s"),
        ],
    )
    meter, tc, tu, tr, ts = render_report(spec, kb)["components"]
    claim = kb.get_claim("c_bind_P1_MC4R")
    assert meter["c"] == round(claim.c, 4)
    assert tc["value"] == round(claim.c, 4) and tc["label"] == "confidence"
    assert tu["value"] == round(claim.u, 4) and tu["label"] == "uncertainty"
    assert tr["value"] == 5 and tr["label"] == "supporting"
    assert ts["value"] == 1 and ts["label"] == "contradicting"


def test_render_sparkline_and_provenance_from_trajectory(kb: KB):
    spec = ReportSpec(
        title="t",
        components=[
            TrajectorySparkline(claim_id="c_bind_P1_MC4R"),
            ProvenanceTrail(claim_id="c_bind_P1_MC4R"),
        ],
    )
    spark, prov = render_report(spec, kb)["components"]
    assert [p["t"] for p in spark["points"]] == [0, 1]
    assert spark["points"][-1]["c"] == round(sigmoid(logit(0.9)), 4)
    # provenance carries the audit causes verbatim, in order
    assert [s["cause"] for s in prov["steps"]] == [
        "seed from prior",
        "APPLY_EVIDENCE from Nature [strong]",
    ]


def test_render_evidence_table_by_tag_and_ids(kb: KB):
    # tag binding pulls every claim carrying peptide:p1, ranked by confidence
    by_tag = render_report(
        ReportSpec(title="t", components=[EvidenceTable(tag="peptide:p1")]), kb
    )["components"][0]
    assert by_tag["grounded"] is True
    assert [r["id"] for r in by_tag["rows"]] == ["c_bind_P1_MC4R", "c_efficacy_P1"]
    # explicit id binding preserves the given order
    by_ids = render_report(
        ReportSpec(
            title="t",
            components=[EvidenceTable(claim_ids=["c_efficacy_P1", "c_bind_P1_MC4R"])],
        ),
        kb,
    )["components"][0]
    assert [r["id"] for r in by_ids["rows"]] == ["c_efficacy_P1", "c_bind_P1_MC4R"]


def test_render_contradiction_panel_from_live_edges(kb: KB):
    out = render_report(ReportSpec(title="t", components=[ContradictionPanel()]), kb)
    panel = out["components"][0]
    assert panel["grounded"] is True
    assert len(panel["pairs"]) == 1
    pair = panel["pairs"][0]
    assert pair["src"]["id"] == "c_bind_P1_MC4R"
    assert pair["dst"]["id"] == "c_efficacy_P1"
    assert pair["src"]["c"] == round(kb.get_claim("c_bind_P1_MC4R").c, 4)


def test_render_metric_comparison(kb: KB):
    out = render_report(
        ReportSpec(
            title="t",
            components=[MetricComparison(claim_ids=["c_bind_P1_MC4R", "c_efficacy_P1"])],
        ),
        kb,
    )
    items = out["components"][0]["items"]
    assert [i["id"] for i in items] == ["c_bind_P1_MC4R", "c_efficacy_P1"]
    assert items[0]["c"] == round(kb.get_claim("c_bind_P1_MC4R").c, 4)


def test_render_missing_claim_is_ungrounded_and_fabricates_nothing(kb: KB):
    spec = ReportSpec(title="t", components=[ClaimCard(claim_id="c_nonexistent")])
    comp = render_report(spec, kb)["components"][0]
    assert comp["grounded"] is False
    assert "c_nonexistent" in comp["error"]
    assert "claim" not in comp  # no fabricated numbers
    assert render_report(spec, kb)["grounded"] is False


def test_render_evidence_table_missing_tag_ungrounded(kb: KB):
    comp = render_report(
        ReportSpec(title="t", components=[EvidenceTable(tag="peptide:zzz")]), kb
    )["components"][0]
    assert comp["grounded"] is False
    assert "rows" not in comp


# --------------------------------------------------------------------------
# groundedness: the output-side immune system.
# --------------------------------------------------------------------------


def test_groundedness_passes_on_valid_spec(kb: KB):
    spec = ReportSpec(
        title="t",
        components=[
            ClaimCard(claim_id="c_bind_P1_MC4R"),
            ProseBlock(text="P1 binds its target with high confidence.", cites=["c_bind_P1_MC4R"]),
            ContradictionPanel(),
        ],
    )
    result = groundedness(spec, kb)
    assert result["ok"] is True
    assert result["violations"] == []


def test_groundedness_catches_bogus_claim_id(kb: KB):
    spec = ReportSpec(title="t", components=[ConfidenceMeter(claim_id="c_ghost")])
    result = groundedness(spec, kb)
    assert result["ok"] is False
    assert result["violations"][0]["component"] == 0
    assert result["violations"][0]["type"] == "confidence_meter"
    assert "c_ghost" in result["violations"][0]["detail"]


def test_groundedness_catches_uncited_prose_block(kb: KB):
    # a prose block that cites nothing …
    empty = groundedness(
        ReportSpec(title="t", components=[ProseBlock(text="Trust me.", cites=[])]), kb
    )
    assert empty["ok"] is False
    assert empty["violations"][0]["type"] == "prose_block"
    # … and one that cites only a non-existent claim
    bogus = groundedness(
        ReportSpec(title="t", components=[ProseBlock(text="Trust me.", cites=["c_ghost"])]), kb
    )
    assert bogus["ok"] is False
    assert bogus["violations"][0]["type"] == "prose_block"


def test_groundedness_flags_partial_missing_metric_ids(kb: KB):
    spec = ReportSpec(
        title="t", components=[MetricComparison(claim_ids=["c_bind_P1_MC4R", "c_ghost"])]
    )
    result = groundedness(spec, kb)
    assert result["ok"] is False
    assert "c_ghost" in result["violations"][0]["detail"]


# --------------------------------------------------------------------------
# default_report: deterministic, fully-grounded fallback / template.
# --------------------------------------------------------------------------


def test_default_report_is_grounded(kb: KB):
    spec = default_report(kb)
    assert isinstance(spec, ReportSpec)
    assert spec.components  # non-empty
    result = groundedness(spec, kb)
    assert result["ok"] is True, result["violations"]
    rendered = render_report(spec, kb)
    assert rendered["grounded"] is True
    # the lead claim is the highest-confidence one
    meter = next(c for c in rendered["components"] if c["type"] == "confidence_meter")
    assert meter["claim_id"] == "c_bind_P1_MC4R"


def test_default_report_empty_kb_is_valid():
    spec = default_report(KB())
    assert isinstance(spec, ReportSpec)
    assert spec.components == []
    assert groundedness(spec, KB())["ok"] is True
    assert render_report(spec, KB())["grounded"] is True
