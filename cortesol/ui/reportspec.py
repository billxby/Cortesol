"""Grounded generative reports (area C) — the LLM proposes the layout; the ledger
supplies the facts.

This is the mission-loyal answer to "generate me a report/UI". A model may only emit
a schema-constrained :class:`ReportSpec`: a title plus a list of components drawn from
a CLOSED vocabulary, where every component is BOUND TO A KB QUERY (a claim id, a tag,
or the graph's live edges) — never to free-floating facts the model typed. A
deterministic renderer (:func:`render_report`) then resolves each component against the
belief graph, reading numbers straight off ``claim.c / u / r / s``, the trajectory, and
``kb.live_edges()``. The report therefore CANNOT assert anything the graph doesn't
support: "truth in, truth out".

The output-side immune system is :func:`groundedness` — it checks that every referenced
claim id / tag exists and that every prose block cites at least one real claim. The
renderer is defensive on its own too: a component that references a missing claim is
marked ``{"grounded": false, "error": ...}`` and renders NOTHING fabricated.

This module is PURE and deterministic: no network, no LLM, no global state. It reads a
:class:`~cortesol.core.kb.KB` and never mutates belief. (Prime Directives PD1/PD6.)

Conversational live-editing of a spec is intentionally OUT OF SCOPE here; the seam for
it is that a ReportSpec is a plain, serializable value — an editor would transform one
spec into another and re-render, never touching belief.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..core.kb import KB
from ..core.mathx import sigmoid
from ..core.schema import Claim

# --------------------------------------------------------------------------
# The component vocabulary — a CLOSED, Literal-tagged discriminated union. A
# component names a *kind of view* and the KB query that fills it; it can never
# carry a free numeric fact. Adding a kind is a deliberate, reviewable act.
# --------------------------------------------------------------------------


class ClaimCard(BaseModel):
    """A claim's headline card: its text, confidence, uncertainty, evidence, status."""

    type: Literal["claim_card"] = "claim_card"
    claim_id: str


class ConfidenceMeter(BaseModel):
    """A single confidence bar for one claim (with its uncertainty band)."""

    type: Literal["confidence_meter"] = "confidence_meter"
    claim_id: str


class TrajectorySparkline(BaseModel):
    """The claim's belief trajectory (ℓ / c over time) as sparkline points."""

    type: Literal["trajectory_sparkline"] = "trajectory_sparkline"
    claim_id: str


class StatTile(BaseModel):
    """One scalar read straight off a claim: confidence, uncertainty, or the
    supporting / contradicting evidence counts."""

    type: Literal["stat_tile"] = "stat_tile"
    claim_id: str
    stat: Literal["c", "u", "r", "s"] = "c"


class EvidenceTable(BaseModel):
    """A table of matching claims (id, text, c, u, r/s). Bound by an explicit list
    of claim ids, an ontology tag, or both — never free rows."""

    type: Literal["evidence_table"] = "evidence_table"
    claim_ids: list[str] | None = None
    tag: str | None = None


class ContradictionPanel(BaseModel):
    """Claims joined by LIVE ``contradicts`` edges — read from the graph itself, so
    it takes no arguments and cannot be spoofed."""

    type: Literal["contradiction_panel"] = "contradiction_panel"


class ProvenanceTrail(BaseModel):
    """The audit chain for one claim: every trajectory step and its cause."""

    type: Literal["provenance_trail"] = "provenance_trail"
    claim_id: str


class MetricComparison(BaseModel):
    """Compare confidence across an explicit set of claims."""

    type: Literal["metric_comparison"] = "metric_comparison"
    claim_ids: list[str]


class ProseBlock(BaseModel):
    """Free prose the model wrote — allowed, but it MUST cite at least one existing
    claim. The citations are what tie the words back to the ledger."""

    type: Literal["prose_block"] = "prose_block"
    text: str
    cites: list[str] = Field(default_factory=list)


Component = Annotated[
    ClaimCard
    | ConfidenceMeter
    | TrajectorySparkline
    | StatTile
    | EvidenceTable
    | ContradictionPanel
    | ProvenanceTrail
    | MetricComparison
    | ProseBlock,
    Field(discriminator="type"),
]


class ReportSpec(BaseModel):
    """The whole proposal: a title, an optional subtitle, and an ordered list of
    components. This is the ONLY thing a model may emit — it proposes structure, not
    facts."""

    title: str
    subtitle: str | None = None
    components: list[Component] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Rendering — resolve every component against the KB into concrete render-data.
# The renderer NEVER invents a value: a number comes from claim.c/u/r/s, a
# sparkline from the trajectory, provenance from trajectory causes, and
# contradictions from live edges. A missing reference renders as an explicit,
# fabricated-nothing error component.
# --------------------------------------------------------------------------


def _claim_row(cl: Claim) -> dict:
    """The canonical read-only view of a claim — every number straight off the ledger."""
    return {
        "id": cl.id,
        "text": cl.text,
        "c": round(cl.c, 4),
        "u": round(cl.u, 4),
        "r": cl.r,
        "s": cl.s,
        "status": cl.status.value,
        "tags": list(cl.ontology_tags),
    }


def _ungrounded(kind: str, error: str, **extra) -> dict:
    """A component that could not be bound to the ledger. Carries NO fabricated
    figures — only the reason it is empty, so the UI can flag it honestly."""
    return {"type": kind, "grounded": False, "error": error, **extra}


def _sparkline_points(cl: Claim) -> list[dict]:
    return [
        {"t": p.t, "ell": round(p.ell, 4), "c": round(sigmoid(p.ell), 4)}
        for p in cl.trajectory
    ]


def _render_claim_card(comp: ClaimCard, kb: KB) -> dict:
    cl = kb.get_claim(comp.claim_id)
    if cl is None:
        return _ungrounded(comp.type, f"unknown claim '{comp.claim_id}'", claim_id=comp.claim_id)
    return {"type": comp.type, "grounded": True, "claim": _claim_row(cl)}


def _render_confidence_meter(comp: ConfidenceMeter, kb: KB) -> dict:
    cl = kb.get_claim(comp.claim_id)
    if cl is None:
        return _ungrounded(comp.type, f"unknown claim '{comp.claim_id}'", claim_id=comp.claim_id)
    return {
        "type": comp.type,
        "grounded": True,
        "claim_id": cl.id,
        "text": cl.text,
        "c": round(cl.c, 4),
        "u": round(cl.u, 4),
        "status": cl.status.value,
    }


def _render_sparkline(comp: TrajectorySparkline, kb: KB) -> dict:
    cl = kb.get_claim(comp.claim_id)
    if cl is None:
        return _ungrounded(comp.type, f"unknown claim '{comp.claim_id}'", claim_id=comp.claim_id)
    return {
        "type": comp.type,
        "grounded": True,
        "claim_id": cl.id,
        "text": cl.text,
        "c": round(cl.c, 4),
        "points": _sparkline_points(cl),
    }


_STAT_LABEL = {"c": "confidence", "u": "uncertainty", "r": "supporting", "s": "contradicting"}


def _render_stat_tile(comp: StatTile, kb: KB) -> dict:
    cl = kb.get_claim(comp.claim_id)
    if cl is None:
        return _ungrounded(
            comp.type, f"unknown claim '{comp.claim_id}'", claim_id=comp.claim_id, stat=comp.stat
        )
    if comp.stat == "c":
        value: float | int = round(cl.c, 4)
    elif comp.stat == "u":
        value = round(cl.u, 4)
    elif comp.stat == "r":
        value = cl.r
    else:
        value = cl.s
    return {
        "type": comp.type,
        "grounded": True,
        "claim_id": cl.id,
        "text": cl.text,
        "stat": comp.stat,
        "label": _STAT_LABEL[comp.stat],
        "value": value,
    }


def _claims_for_table(comp: EvidenceTable, kb: KB) -> list[Claim]:
    """Resolve an evidence table's binding to concrete claims: explicit ids first
    (in order, existing only), then any claim carrying the tag, de-duplicated."""
    seen: set[str] = set()
    out: list[Claim] = []
    for cid in comp.claim_ids or []:
        cl = kb.get_claim(cid)
        if cl is not None and cl.id not in seen:
            seen.add(cl.id)
            out.append(cl)
    if comp.tag:
        for cl in sorted(kb.claims.values(), key=lambda c: (-c.c, c.id)):
            if comp.tag in cl.ontology_tags and cl.id not in seen:
                seen.add(cl.id)
                out.append(cl)
    return out


def _render_evidence_table(comp: EvidenceTable, kb: KB) -> dict:
    rows = [_claim_row(cl) for cl in _claims_for_table(comp, kb)]
    if not rows:
        detail = (
            f"no claims match tag '{comp.tag}'"
            if comp.tag
            else "no matching claim ids in the ledger"
        )
        return _ungrounded(comp.type, detail, tag=comp.tag, claim_ids=comp.claim_ids or [])
    return {"type": comp.type, "grounded": True, "tag": comp.tag, "rows": rows}


def _render_contradiction_panel(comp: ContradictionPanel, kb: KB) -> dict:
    pairs = []
    for e in kb.live_edges():
        if e.type.value != "contradicts":
            continue
        src, dst = kb.get_claim(e.src), kb.get_claim(e.dst)
        if src is None or dst is None:
            continue
        pairs.append(
            {
                "src": {"id": src.id, "text": src.text, "c": round(src.c, 4)},
                "dst": {"id": dst.id, "text": dst.text, "c": round(dst.c, 4)},
                "weight": round(e.weight, 3),
            }
        )
    # Structurally grounded regardless of count — it reads the graph, not a reference.
    return {"type": comp.type, "grounded": True, "pairs": pairs}


def _render_provenance_trail(comp: ProvenanceTrail, kb: KB) -> dict:
    cl = kb.get_claim(comp.claim_id)
    if cl is None:
        return _ungrounded(comp.type, f"unknown claim '{comp.claim_id}'", claim_id=comp.claim_id)
    steps = [
        {"t": p.t, "ell": round(p.ell, 4), "c": round(sigmoid(p.ell), 4), "cause": p.cause}
        for p in cl.trajectory
    ]
    return {
        "type": comp.type,
        "grounded": True,
        "claim_id": cl.id,
        "text": cl.text,
        "steps": steps,
    }


def _render_metric_comparison(comp: MetricComparison, kb: KB) -> dict:
    items = []
    for cid in comp.claim_ids:
        cl = kb.get_claim(cid)
        if cl is not None:
            items.append({"id": cl.id, "text": cl.text, "c": round(cl.c, 4), "u": round(cl.u, 4)})
    if not items:
        return _ungrounded(
            comp.type, "no matching claim ids in the ledger", claim_ids=comp.claim_ids
        )
    return {"type": comp.type, "grounded": True, "items": items}


def _render_prose_block(comp: ProseBlock, kb: KB) -> dict:
    cites = []
    for cid in comp.cites:
        cl = kb.get_claim(cid)
        if cl is not None:
            cites.append({"id": cl.id, "text": cl.text, "c": round(cl.c, 4)})
    if not cites:
        # Prose with no resolvable citation is exactly what "truth in, truth out"
        # forbids — render the words but flag them as unsupported (never hidden).
        return {
            "type": comp.type,
            "grounded": False,
            "error": "prose cites no existing claim",
            "text": comp.text,
            "cites": [],
        }
    return {"type": comp.type, "grounded": True, "text": comp.text, "cites": cites}


_RENDERERS = {
    "claim_card": _render_claim_card,
    "confidence_meter": _render_confidence_meter,
    "trajectory_sparkline": _render_sparkline,
    "stat_tile": _render_stat_tile,
    "evidence_table": _render_evidence_table,
    "contradiction_panel": _render_contradiction_panel,
    "provenance_trail": _render_provenance_trail,
    "metric_comparison": _render_metric_comparison,
    "prose_block": _render_prose_block,
}


def render_report(spec: ReportSpec, kb: KB) -> dict:
    """Resolve a :class:`ReportSpec` against the belief graph into concrete
    render-data. Every figure traces to ``kb``; nothing is fabricated. A component
    that references a missing claim / tag comes back ``grounded: false`` with an
    error and no invented values."""
    components = [_RENDERERS[c.type](c, kb) for c in spec.components]
    return {
        "title": spec.title,
        "subtitle": spec.subtitle,
        "components": components,
        "grounded": all(c.get("grounded", False) for c in components) if components else True,
    }


# --------------------------------------------------------------------------
# Groundedness — the OUTPUT-side immune system. It answers one question: does
# every claim the spec references actually exist in the ledger, and does every
# prose block cite at least one real claim? Violations are reported, not hidden.
# --------------------------------------------------------------------------

_SINGLE_CLAIM_KINDS = {
    "claim_card",
    "confidence_meter",
    "trajectory_sparkline",
    "stat_tile",
    "provenance_trail",
}


def groundedness(spec: ReportSpec, kb: KB) -> dict:
    """Check that the spec is fully bound to the ledger. Returns
    ``{"ok": bool, "violations": [...]}`` where each violation names the offending
    component index, its type, and why it fails. ``ok`` is True only when the report
    could be rendered with every figure traceable to a real claim."""
    violations: list[dict] = []

    def flag(i: int, kind: str, detail: str) -> None:
        violations.append({"component": i, "type": kind, "detail": detail})

    for i, comp in enumerate(spec.components):
        if comp.type in _SINGLE_CLAIM_KINDS:
            if kb.get_claim(comp.claim_id) is None:  # type: ignore[attr-defined]
                flag(i, comp.type, f"references unknown claim '{comp.claim_id}'")  # type: ignore[attr-defined]
        elif comp.type == "metric_comparison":
            missing = [cid for cid in comp.claim_ids if kb.get_claim(cid) is None]
            if not comp.claim_ids:
                flag(i, comp.type, "no claim ids to compare")
            elif missing:
                flag(i, comp.type, f"references unknown claim(s): {', '.join(missing)}")
        elif comp.type == "evidence_table":
            if not comp.claim_ids and not comp.tag:
                flag(i, comp.type, "bound to neither claim ids nor a tag")
                continue
            missing = [cid for cid in (comp.claim_ids or []) if kb.get_claim(cid) is None]
            if missing:
                flag(i, comp.type, f"references unknown claim(s): {', '.join(missing)}")
            if comp.tag and not any(comp.tag in c.ontology_tags for c in kb.claims.values()):
                flag(i, comp.type, f"tag '{comp.tag}' matches no claim")
        elif comp.type == "prose_block":
            existing = [cid for cid in comp.cites if kb.get_claim(cid) is not None]
            if not existing:
                flag(i, comp.type, "prose block cites no existing claim (>=1 required)")
        # contradiction_panel takes no reference — always grounded.

    return {"ok": not violations, "violations": violations}


# --------------------------------------------------------------------------
# Default report — a deterministic, fully-grounded ReportSpec built from the KB.
# Doubles as the no-LLM fallback AND as a worked template for the model.
# --------------------------------------------------------------------------


def _top_claims(kb: KB, k: int) -> list[Claim]:
    return sorted(kb.claims.values(), key=lambda c: (-c.c, c.id))[:k]


def default_report(kb: KB, k: int = 6) -> ReportSpec:
    """Build a fully-grounded ReportSpec from the top-``k`` highest-confidence claims.
    Used as the offline fallback (no key / any LLM failure) and as a template the
    model can imitate. Empty KB -> a valid, empty-but-honest report."""
    top = _top_claims(kb, k)
    if not top:
        return ReportSpec(
            title="Belief report",
            subtitle="The belief graph holds no claims yet — ingest evidence to populate it.",
            components=[],
        )

    lead = top[0]
    ids = [c.id for c in top]
    components: list[Component] = [
        ProseBlock(
            text=(
                "This report is generated from the belief graph: every figure below is read "
                "straight off the ledger, never asserted by a language model. The most "
                "confident claim on record is highlighted first."
            ),
            cites=[lead.id],
        ),
        ConfidenceMeter(claim_id=lead.id),
        TrajectorySparkline(claim_id=lead.id),
        ProvenanceTrail(claim_id=lead.id),
        EvidenceTable(claim_ids=ids),
    ]
    if len(top) >= 2:
        components.append(MetricComparison(claim_ids=ids[: min(4, len(ids))]))
    components.append(ContradictionPanel())
    plural = "s" if len(top) != 1 else ""
    return ReportSpec(
        title="Belief report",
        subtitle=f"Top {len(top)} claim{plural} by confidence, straight from the ledger.",
        components=components,
    )


def compact_kb_summary(kb: KB, claim_ids: list[str] | None = None, limit: int = 60) -> str:
    """A compact, model-facing digest of the ledger: one line per claim as
    ``id | text | c=… | tags``. This is the ONLY set of claim ids a model may
    reference; it is DATA handed to the model, never instructions. Claims are the
    given ids (if any) else all claims, ranked by confidence and capped at ``limit``."""
    if claim_ids is not None:
        claims = [kb.claims[cid] for cid in claim_ids if cid in kb.claims]
    else:
        claims = list(kb.claims.values())
    claims = sorted(claims, key=lambda c: (-c.c, c.id))[:limit]
    lines = []
    for cl in claims:
        tags = ",".join(cl.ontology_tags) if cl.ontology_tags else "-"
        text = cl.text.replace("\n", " ").strip()
        if len(text) > 160:
            text = text[:157] + "…"
        lines.append(
            f"{cl.id} | {text} | c={round(cl.c, 3)} | status={cl.status.value} | tags={tags}"
        )
    return "\n".join(lines)
