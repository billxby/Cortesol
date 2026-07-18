"""Graph propagation + the retraction cascade.

Logical area A (Ground-Truth Engine). Deterministic, no LLM.
Reference: Research/"Graph Propagation and GNNs" §1-2, "Belief Revision" §TMS.

TruthFinder source<->claim loop + typed belief propagation over the dirty k-hop
neighborhood (config.EDGE_INFLUENCE / BP_* knobs). Discrediting a source sinks
everything it supported — the demo money shot plain RAG can't do.
"""

from __future__ import annotations

from . import config
from .kb import KB
from .mathx import clip, logit, sigmoid
from .results import Delta
from .schema import ClaimStatus


def propagate(kb: KB, dirty: set[str]) -> list[Delta]:
    """Recompute dependent claims in the dirty neighborhood; return ripple Deltas."""
    nodes = sorted(cid for cid in dirty if cid in kb.claims)
    if not nodes:
        return []
    current = {cid: kb.claims[cid].ell for cid in nodes}
    estimates = dict(current)
    live = kb.live_edges()

    for _ in range(config.BP_MAX_ITERS):
        updated = dict(estimates)
        max_change = 0.0
        for cid in nodes:
            influence = 0.0
            for edge in live:
                other: str | None = None
                if edge.dst == cid:
                    other = edge.src
                elif edge.src == cid and edge.type.value != "depends_on":
                    other = edge.dst
                if other is None or other not in kb.claims:
                    continue
                other_ell = estimates.get(other, kb.claims[other].ell)
                centered = 2.0 * (sigmoid(other_ell) - 0.5)
                influence += (
                    config.TRUTHFINDER_RHO
                    * config.EDGE_INFLUENCE[edge.type.value]
                    * edge.weight
                    * centered
                )
            target = current[cid] + influence
            value = config.BP_DAMPING * target + (1.0 - config.BP_DAMPING) * estimates[cid]
            updated[cid] = value
            max_change = max(max_change, abs(value - estimates[cid]))
        estimates = updated
        if max_change < 1e-6:
            break

    deltas: list[Delta] = []
    for cid in nodes:
        claim = kb.claims[cid]
        if claim.status in {ClaimStatus.QUARANTINED, ClaimStatus.RETIRED}:
            continue
        movement = clip(estimates[cid] - claim.ell, -config.DELTA_MAX, config.DELTA_MAX)
        if abs(movement) < 1e-9:
            continue
        before = claim.ell
        kb.move_belief(cid, movement, cause="typed-belief-propagation")
        deltas.append(
            Delta(
                claim_id=cid,
                before_ell=before,
                after_ell=claim.ell,
                cause="typed-belief-propagation",
            )
        )
    return deltas


def discredit_source(kb: KB, source_id: str) -> list[Delta]:
    """Retraction cascade: flag the source, propagate the un-belief downstream."""
    source = kb.get_source(source_id)
    if source is None:
        raise KeyError(f"unknown source: {source_id}")
    source.discredited = True
    direct: list[Delta] = []
    evidence_ids = set(source.history)
    dirty: set[str] = set()
    for claim in kb.claims.values():
        prior = logit(config.PRIOR_C_0)
        attributed = 0.0
        previous = prior
        for point in claim.trajectory:
            step = point.ell - previous
            if any(eid in point.cause for eid in evidence_ids):
                attributed += step
            previous = point.ell
        reversal = clip(-attributed, -config.DELTA_MAX, config.DELTA_MAX)
        if abs(reversal) < 1e-9:
            continue
        before = claim.ell
        kb.move_belief(claim.id, reversal, cause=f"source-discredited:{source_id}")
        direct.append(
            Delta(
                claim_id=claim.id,
                before_ell=before,
                after_ell=claim.ell,
                cause=f"source-discredited:{source_id}",
            )
        )
        dirty.add(claim.id)
    return [*direct, *propagate(kb, kb.dirty_neighborhood(dirty))]
