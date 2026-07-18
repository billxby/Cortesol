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
import re

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
from .schema import Claim, Edge, EdgeType, Evidence

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def claim_id_for(text: str) -> str:
    """Deterministic claim id from proposition text, so the same ADD_CLAIM never
    creates a duplicate node. Pure function — no randomness (core stays reproducible)."""
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return f"c_{slug[:64]}" if slug else "c_unnamed"


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


def _lambda_breakdown(kb: KB, op: ApplyEvidence, evidence: Evidence) -> dict:
    """Pure, NON-MUTATING computation of the capped/damped Λ and the signed Δℓ for
    one APPLY_EVIDENCE op, keeping every intermediate quantity. This is the single
    source of the update arithmetic: `apply_evidence` calls it and then commits the
    `move_belief` + bookkeeping, while `explain_apply_evidence` calls it read-only to
    render the step-by-step trace for the audit log / UI. Nothing here writes state
    (in particular it does NOT bump `correlation_seen` — the caller does)."""
    claim = kb.claims[op.claim_id]
    source = kb.get_source(evidence.source_id)

    lam_raw = strength_to_loglr(op.strength)
    cap = source_cap(source.tau, source.phi) if source is not None else None
    lam_capped = min(lam_raw, cap) if cap is not None else lam_raw

    group = evidence.correlation_group or correlation_group(evidence.fields)
    k_index = claim.correlation_seen.get(group, 0)
    neff = neff_marginal_factor(k_index, RHO_WITHIN_GROUP)
    lam_after_neff = lam_capped * neff

    phi = source.phi if source is not None else 0.30
    fraud = fraud_switch_factor(evidence.red_flags, phi)
    lam_after_fraud = lam_after_neff * fraud

    delta = clip(lam_after_fraud, -DELTA_MAX, DELTA_MAX)
    signed = delta if op.direction == "+" else -delta

    return {
        "claim": claim,
        "source": source,
        "group": group,
        "k_index": k_index,
        "phi": phi,
        "lam_raw": lam_raw,
        "cap": cap,
        "lam_capped": lam_capped,
        "neff": neff,
        "lam_after_neff": lam_after_neff,
        "fraud": fraud,
        "lam_after_fraud": lam_after_fraud,
        "delta": delta,
        "signed": signed,
    }


def apply_evidence(kb: KB, op: ApplyEvidence, evidence: Evidence) -> Delta:
    """Apply one APPLY_EVIDENCE op to the KB and return the attributed Delta."""
    b = _lambda_breakdown(kb, op, evidence)
    claim = b["claim"]
    source = b["source"]

    # commit the n_eff bookkeeping the breakdown deliberately left untouched, so the
    # next correlated echo in this group is damped further.
    claim.correlation_seen[b["group"]] = b["k_index"] + 1

    before = claim.ell
    tier = source.tier if source is not None else "unknown"
    cause = (
        f"APPLY_EVIDENCE {op.direction}{op.strength} from {evidence.source_id} "
        f"[{tier}] (ev={evidence.id})"
    )
    kb.move_belief(op.claim_id, b["signed"], cause=cause)

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


def explain_apply_evidence(kb: KB, op: ApplyEvidence, evidence: Evidence) -> dict:
    """Read-only 'show your work' trace for one APPLY_EVIDENCE op: the ordered
    quantization from the model's coarse strength label to the committed Δℓ, using
    the SAME arithmetic as `apply_evidence` (via `_lambda_breakdown`) but WITHOUT
    moving belief or touching any bookkeeping. Consumed by the UI reasoning log so a
    judge can expand any action and see every capped/damped step. Call it BEFORE
    `apply_evidence` commits, so the pre-update ℓ and echo index are the real ones."""
    from .mathx import sigmoid

    b = _lambda_breakdown(kb, op, evidence)
    claim = b["claim"]
    source = b["source"]
    before_ell = claim.ell
    after_ell = before_ell + b["signed"]

    steps = [
        {
            "label": "strength → logΛ",
            "detail": f"{op.strength} → {b['lam_raw']:.3f}",
            "note": "coarse evidence-quality label mapped to a pre-cap log-likelihood ratio",
        },
    ]
    if b["cap"] is not None:
        tier = source.tier if source is not None else "unknown"
        capped = "cap binds" if b["lam_capped"] < b["lam_raw"] else "under cap"
        steps.append(
            {
                "label": "source cap",
                "detail": f"min({b['lam_raw']:.3f}, log(τ/φ)={b['cap']:.3f}) = {b['lam_capped']:.3f}",
                "note": f"anti-hype ceiling for a '{tier}' source (τ={source.tau}, φ={source.phi}) — {capped}",
            }
        )
    steps.append(
        {
            "label": "× n_eff",
            "detail": f"× {b['neff']:.3f} = {b['lam_after_neff']:.3f}",
            "note": (
                "first report in its correlation group — full weight"
                if b["k_index"] == 0
                else f"echo #{b['k_index'] + 1} in group '{b['group']}' — correlated, damped"
            ),
        }
    )
    steps.append(
        {
            "label": "× fraud switch",
            "detail": f"× {b['fraud']:.3f} = {b['lam_after_fraud']:.3f}",
            "note": (
                "no red flags — Λ intact"
                if not evidence.red_flags
                else f"red flags {list(evidence.red_flags)} raise φ_eff, shrinking Λ"
            ),
        }
    )
    if abs(b["lam_after_fraud"]) > DELTA_MAX:
        steps.append(
            {
                "label": "clamp ±DELTA_MAX",
                "detail": f"clip({b['lam_after_fraud']:.3f}, ±{DELTA_MAX}) = {b['delta']:.3f}",
                "note": "per-event blast-radius bound (PD2)",
            }
        )
    steps.append(
        {
            "label": "apply direction",
            "detail": f"{op.direction} → Δℓ = {b['signed']:+.3f}",
            "note": "'+' supports the claim, '−' contradicts it",
        }
    )
    steps.append(
        {
            "label": "move belief",
            "detail": (
                f"ℓ {before_ell:+.3f} → {after_ell:+.3f}  "
                f"(c {sigmoid(before_ell):.3f} → {sigmoid(after_ell):.3f})"
            ),
            "note": "the only belief mutation — through the engine, never set directly",
        }
    )

    return {
        "claim_id": op.claim_id,
        "direction": op.direction,
        "strength": op.strength,
        "evidence_id": op.evidence_id,
        "before_ell": round(before_ell, 4),
        "after_ell": round(after_ell, 4),
        "c_before": round(sigmoid(before_ell), 4),
        "c_after": round(sigmoid(after_ell), 4),
        "d_c": round(sigmoid(after_ell) - sigmoid(before_ell), 4),
        "steps": steps,
    }


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

    if isinstance(op, AddClaim):
        # Materialise a new proposition at the skeptical prior with high u. This is
        # a STRUCTURAL add (like ADD_EDGE) — it introduces a node, it never *sets*
        # belief. Movement still only ever happens via kb.move_belief. (PD2.)
        cid = claim_id_for(op.text)
        if cid not in kb.claims:
            kb.add_claim(Claim(id=cid, text=op.text, ontology_tags=list(op.ontology_tags)))
        return []

    if isinstance(op, (FlagOOD, Reject)):
        # FLAG_OOD and REJECT are record-only refusals — no belief moves. (PD2/PD7.)
        return []

    return []
