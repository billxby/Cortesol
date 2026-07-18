"""Graph propagation + the retraction cascade.

Logical area A (Ground-Truth Engine). Deterministic, no LLM.
Reference: Research/"Graph Propagation and GNNs" §1-2, "Belief Revision" §TMS.

Typed belief propagation over the dirty k-hop neighborhood (config.EDGE_INFLUENCE /
BP_* knobs). Propagation is a *ripple*: a change on one claim is transmitted,
attenuated and sign-corrected, to its typed neighbors — it never overwrites a
claim's own evidence, only echoes changes across relations. Discrediting a source
subtracts everything it ever contributed and lets that un-belief ripple downstream
— the demo money shot plain RAG can't do.
"""

from __future__ import annotations

from .config import (
    BP_DAMPING,
    DELTA_MAX,
    EDGE_INFLUENCE,
    PRIOR_C_0,
    PROPAGATE_KHOP,
)
from .kb import KB
from .mathx import clip, logit
from .results import Delta
from .schema import Edge

_ELL0 = logit(PRIOR_C_0)
_EPS = 1e-4  # below this a ripple is not worth an audit line


def _neighbors(kb: KB, u: str) -> list[tuple[str, Edge]]:
    """Live typed edges incident to claim `u`, paired with the claim they carry the
    ripple to. `depends_on` is directional (src -> dst only, the dependent is dst);
    supports/replicates/contradicts are symmetric."""
    out: list[tuple[str, Edge]] = []
    for e in kb.edges.values():
        if not e.live:
            continue
        if e.type.value == "depends_on":
            if e.src == u:
                out.append((e.dst, e))
        else:
            if e.src == u:
                out.append((e.dst, e))
            elif e.dst == u:
                out.append((e.src, e))
    return out


def _ripple(kb: KB, seed_deltas: dict[str, float], cause: str) -> list[Delta]:
    """Push attenuated, sign-correct echoes of `seed_deltas` (claim_id -> Δell just
    applied) across typed edges, up to PROPAGATE_KHOP hops. Bounded by DELTA_MAX
    per step and decaying by BP_DAMPING x |EDGE_INFLUENCE| < 1 per hop."""
    deltas: list[Delta] = []
    frontier = dict(seed_deltas)
    for _ in range(PROPAGATE_KHOP):
        nxt: dict[str, float] = {}
        for u, du in sorted(frontier.items()):
            if abs(du) < _EPS:
                continue
            for v, edge in _neighbors(kb, u):
                coef = EDGE_INFLUENCE.get(edge.type.value, 0.0) * edge.weight * BP_DAMPING
                dv = clip(coef * du, -DELTA_MAX, DELTA_MAX)
                if abs(dv) < _EPS:
                    continue
                claim = kb.get_claim(v)
                if claim is None:
                    continue
                before = claim.ell
                kb.move_belief(v, dv, cause=f"{cause} via {edge.type.value} from {u}")
                deltas.append(
                    Delta(
                        claim_id=v,
                        before_ell=before,
                        after_ell=claim.ell,
                        cause=f"{cause} ({edge.type.value} <- {u})",
                        op=None,
                    )
                )
                nxt[v] = nxt.get(v, 0.0) + dv
        frontier = nxt
    return deltas


def propagate(kb: KB, dirty: set[str]) -> list[Delta]:
    """Recompute dependent claims in the dirty neighborhood; return ripple Deltas.

    Seeds are the claims in `dirty` that actually moved on the current event
    (their latest trajectory point is stamped with kb.event_cursor). Their move is
    rippled out to typed neighbors."""
    seeds: dict[str, float] = {}
    for cid in dirty:
        claim = kb.get_claim(cid)
        if claim is None or not claim.trajectory:
            continue
        last = claim.trajectory[-1]
        if last.t != kb.event_cursor:
            continue
        prev_ell = claim.trajectory[-2].ell if len(claim.trajectory) >= 2 else _ELL0
        move = last.ell - prev_ell
        if abs(move) >= _EPS:
            seeds[cid] = move
    if not seeds:
        return []
    return _ripple(kb, seeds, cause="propagate")


def _source_contribution(kb: KB, claim_id: str, source_id: str) -> float:
    """How much net Δell this source ever pushed onto this claim, reconstructed
    from the audit trajectory (each APPLY_EVIDENCE cause names its source_id)."""
    claim = kb.claims[claim_id]
    prev = _ELL0
    total = 0.0
    for pt in claim.trajectory:
        delta = pt.ell - prev
        if f"from {source_id} " in pt.cause or pt.cause.endswith(f"from {source_id}"):
            total += delta
        prev = pt.ell
    return total


def discredit_source(kb: KB, source_id: str) -> list[Delta]:
    """Retraction cascade: flag the source, undo every belief increment it ever
    contributed, then ripple that un-belief downstream. (The money shot.)"""
    source = kb.get_source(source_id)
    if source is not None:
        if source.discredited:
            return []  # already retracted — idempotent, don't double-count
        source.discredited = True

    seed_deltas: dict[str, float] = {}
    deltas: list[Delta] = []
    for cid, claim in sorted(kb.claims.items()):
        contribution = _source_contribution(kb, cid, source_id)
        if abs(contribution) < _EPS:
            continue
        before = claim.ell
        kb.move_belief(cid, -contribution, cause=f"discredit {source_id}: retract contribution")
        # retracting positive support removes an r; negative support removes an s
        if contribution > 0 and claim.r > 0:
            claim.r -= 1
        elif contribution < 0 and claim.s > 0:
            claim.s -= 1
        deltas.append(
            Delta(
                claim_id=cid,
                before_ell=before,
                after_ell=claim.ell,
                cause=f"discredit {source_id}: retract {contribution:+.2f}",
                op="DISCREDIT",
            )
        )
        seed_deltas[cid] = -contribution

    deltas.extend(_ripple(kb, seed_deltas, cause=f"discredit {source_id}"))
    return deltas
