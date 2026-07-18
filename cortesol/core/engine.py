"""The belief engine — log-odds updates with source caps and n_eff damping.

OWNER: Branch 1 (ground-truth). Consumes the frozen contracts; implements the
arithmetic in Research/"Confidence Math" §1-3. NO LLM, NO network, deterministic.

Invariant this module upholds (checked by the validator upstream AND asserted
here as defense in depth): a single event moves a single claim by at most
DELTA_MAX in |Δell|, AFTER the source cap and n_eff damping. (PD2.)

    lam  = strength_to_loglr(op.strength)          # coarse label -> log-Lambda
    lam  = min(lam, source_cap(tau, phi))          # §2 anti-hype cap
    lam *= neff_factor(kb, claim_id, evidence)      # §3 echo damping
    lam *= fraud_switch_factor(red_flags, phi)      # §2 fraud/hype self-discount
    delta  = clip(lam, -DELTA_MAX, DELTA_MAX)        # PD2 blast-radius bound
    signed = +delta if direction == '+' else -delta
    kb.move_belief(claim_id, signed, cause)          # the ONLY belief mutator
"""

from __future__ import annotations

import math

from .config import (
    DELTA_MAX,
    PHI_CEILING,
    RED_FLAG_PHI_BUMP,
    RHO_WITHIN_GROUP,
    STRENGTH_TO_LOGLR,
)
from .domain import PEPTIDE_RED_FLAG_PHI_BUMP, correlation_group
from .kb import KB
from .mathx import clip, neff_marginal_factor
from .ops import AddClaim, AddEdge, ApplyEvidence, FlagOOD, InvalidateEdge, Op, Reject
from .results import Delta
from .schema import Edge, EdgeType, Evidence


def strength_to_loglr(strength: str) -> float:
    """Map weak/moderate/strong -> pre-cap log-Lambda (config.STRENGTH_TO_LOGLR)."""
    return STRENGTH_TO_LOGLR[strength]


def source_cap(tau: float, phi: float) -> float:
    """|log Lambda_R| <= log(tau/phi). The anti-hype guarantee (Confidence Math §2).
    A weak source (tau~phi) yields a cap near 0: sensational content, tiny move."""
    return math.log(tau / phi)


def fraud_switch_factor(red_flags: list[str], phi: float) -> float:
    """Return a multiplier in [0, 1] that shrinks Lambda as red flags raise the
    effective phi (Confidence Math §2 mixture; bumps in config + domain).

    Each fired flag adds its phi bump; the accumulated bump is expressed as a
    fraction of the head-room to the phi ceiling, so a battery of serious flags
    (fraud, no control, low purity...) collapses Lambda toward zero while a lone
    soft flag barely dents it.
    """
    bump = 0.0
    for flag in red_flags:
        bump += RED_FLAG_PHI_BUMP.get(flag, PEPTIDE_RED_FLAG_PHI_BUMP.get(flag, 0.0))
    headroom = max(1e-9, PHI_CEILING - max(0.0, phi - 0.30))
    return clip(1.0 - bump / headroom, 0.0, 1.0)


def neff_factor(kb: KB, claim_id: str, evidence: Evidence) -> float:
    """Marginal n_eff weight for this evidence given how many correlated reports
    (same correlation_group) the claim has already absorbed (mathx.neff_marginal_factor).
    First report in a group -> 1.0; echoes -> shrinking. Updates claim.correlation_seen."""
    group = evidence.correlation_group or correlation_group(evidence.fields)
    claim = kb.claims[claim_id]
    k_index = claim.correlation_seen.get(group, 0)
    factor = neff_marginal_factor(k_index, RHO_WITHIN_GROUP)
    claim.correlation_seen[group] = k_index + 1
    return factor


def apply_evidence(kb: KB, op: ApplyEvidence, evidence: Evidence) -> Delta:
    """Apply one APPLY_EVIDENCE op to the KB and return the attributed Delta."""
    claim = kb.claims[op.claim_id]
    source = kb.get_source(evidence.source_id)

    lam = strength_to_loglr(op.strength)
    if source is not None:
        lam = min(lam, source_cap(source.tau, source.phi))
    lam *= neff_factor(kb, op.claim_id, evidence)
    phi = source.phi if source is not None else 0.30
    lam *= fraud_switch_factor(evidence.red_flags, phi)

    delta = clip(lam, -DELTA_MAX, DELTA_MAX)
    signed = delta if op.direction == "+" else -delta

    before = claim.ell
    tier = source.tier if source is not None else "unknown"
    cause = (
        f"APPLY_EVIDENCE {op.direction}{op.strength} from {evidence.source_id} "
        f"[{tier}] (ev={evidence.id})"
    )
    kb.move_belief(op.claim_id, signed, cause=cause)

    if op.direction == "+":
        claim.r += 1
    else:
        claim.s += 1

    return Delta(
        claim_id=op.claim_id,
        before_ell=before,
        after_ell=claim.ell,
        cause=cause,
        op="APPLY_EVIDENCE",
    )


def apply(kb: KB, op: Op, evidence: Evidence | None) -> list[Delta]:
    """Dispatch a single validated op to its handler. APPLY_EVIDENCE moves belief;
    ADD_CLAIM/ADD_EDGE/INVALIDATE_EDGE mutate structure; FLAG_OOD/REJECT record
    without moving belief. Returns the Deltas produced (possibly empty)."""
    if isinstance(op, ApplyEvidence):
        assert evidence is not None, "APPLY_EVIDENCE requires provenance"
        return [apply_evidence(kb, op, evidence)]

    if isinstance(op, AddEdge):
        edge_id = f"edge:{op.src}->{op.dst}:{op.type.value}"
        if edge_id not in kb.edges:
            kb.add_edge(
                Edge(
                    id=edge_id,
                    src=op.src,
                    dst=op.dst,
                    type=EdgeType(op.type),
                    weight=op.weight,
                    valid_at=kb.event_cursor,
                )
            )
        return []

    if isinstance(op, InvalidateEdge):
        if op.edge_id in kb.edges:
            kb.invalidate_edge(op.edge_id)
        return []

    if isinstance(op, (AddClaim, FlagOOD, Reject)):
        # Structural add is handled by the pipeline/validator seeding; FLAG_OOD and
        # REJECT are record-only refusals — no belief moves. (PD2/PD7.)
        return []

    return []
