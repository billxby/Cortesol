"""The belief engine — log-odds updates with source caps and n_eff damping.

OWNER: Branch 1 (ground-truth). Consumes the frozen contracts; implements the
arithmetic in Research/"Confidence Math" §1-3. NO LLM, NO network, deterministic.

This is a STUB. The reference recipe is in the docstrings and Confidence Math
§engine-pseudocode. Implement, then make tests/unit/test_engine.py pass.

Invariant this module must uphold (checked by the validator upstream AND asserted
here as defense in depth): a single event moves a single claim by at most
DELTA_MAX in |Δell|, AFTER the source cap and n_eff damping. (PD2.)
"""

from __future__ import annotations

import hashlib
import json
import math

from . import config, domain
from .kb import KB
from .mathx import clip, neff_marginal_factor
from .ops import AddClaim, AddEdge, ApplyEvidence, InvalidateEdge, Op
from .results import Delta
from .schema import Claim, Edge, Evidence


def strength_to_loglr(strength: str) -> float:
    """Map weak/moderate/strong -> pre-cap log-Lambda (config.STRENGTH_TO_LOGLR)."""
    try:
        return float(config.STRENGTH_TO_LOGLR[strength])
    except KeyError as exc:
        raise ValueError(f"unknown evidence strength: {strength!r}") from exc


def source_cap(tau: float, phi: float) -> float:
    """|log Lambda_R| <= log(tau/phi). The anti-hype guarantee (Confidence Math §2).
    A weak source (tau~phi) yields a cap near 0: sensational content, tiny move."""
    if not (0.0 < phi < tau <= 1.0):
        raise ValueError("source reliability must satisfy 0 < phi < tau <= 1")
    return math.log(tau / phi)


def fraud_switch_factor(red_flags: list[str], phi: float) -> float:
    """Return a multiplier in [0, 1] that shrinks Lambda as red flags raise the
    effective phi (Confidence Math §2 mixture; bumps in config + domain)."""
    bumps = {**config.RED_FLAG_PHI_BUMP, **domain.PEPTIDE_RED_FLAG_PHI_BUMP}
    phi_eff = min(config.PHI_CEILING, phi + sum(bumps.get(flag, 0.0) for flag in set(red_flags)))
    available = config.PHI_CEILING - phi
    if available <= 0.0:
        return 0.0
    return clip((config.PHI_CEILING - phi_eff) / available, 0.0, 1.0)


def neff_factor(kb: KB, claim_id: str, evidence: Evidence) -> float:
    """Marginal n_eff weight for this evidence given how many correlated reports
    (same correlation_group) the claim has already absorbed (mathx.neff_marginal_factor).
    First report in a group -> 1.0; echoes -> shrinking. Updates claim.correlation_seen."""
    claim = kb.get_claim(claim_id)
    if claim is None:
        raise KeyError(f"unknown claim: {claim_id}")
    group = evidence.correlation_group
    if not group:
        return 1.0
    seen = claim.correlation_seen.get(group, 0)
    claim.correlation_seen[group] = seen + 1
    return neff_marginal_factor(seen, config.RHO_WITHIN_GROUP)


def apply_evidence(kb: KB, op: ApplyEvidence, evidence: Evidence) -> Delta:
    """Apply one APPLY_EVIDENCE op to the KB and return the attributed Delta.

    Recipe (Confidence Math §engine-pseudocode):
        lam  = strength_to_loglr(op.strength)
        lam  = min(lam, source_cap(source.tau, source.phi))     # §2 cap
        lam *= neff_factor(kb, op.claim_id, evidence)           # §3 damping
        lam *= fraud_switch_factor(evidence.red_flags, phi)     # §2 mixture
        delta = clip(lam, -DELTA_MAX, DELTA_MAX)                # PD2 bound
        signed = delta if op.direction == '+' else -delta
        kb.move_belief(op.claim_id, signed, cause=...)          # updates trajectory
        update (r, s) counters on the claim                     # §4
    Return a Delta with before/after ell for the audit log.
    """
    claim = kb.get_claim(op.claim_id)
    source = kb.get_source(evidence.source_id)
    if claim is None:
        raise KeyError(f"unknown claim: {op.claim_id}")
    if source is None:
        raise KeyError(f"unknown source: {evidence.source_id}")
    if op.evidence_id != evidence.id:
        raise ValueError("operation evidence_id does not match current evidence")

    before = claim.ell
    lam = min(strength_to_loglr(op.strength), source_cap(source.tau, source.phi))
    lam *= neff_factor(kb, op.claim_id, evidence)
    lam *= fraud_switch_factor(evidence.red_flags, source.phi)
    delta = clip(lam, 0.0, config.DELTA_MAX)
    signed = delta if op.direction == "+" else -delta
    cause = f"{op.op}:{evidence.id}:source={source.id}"
    kb.move_belief(op.claim_id, signed, cause=cause)
    if op.direction == "+":
        claim.r += 1
    else:
        claim.s += 1
    if evidence.id not in source.history:
        source.history.append(evidence.id)
    return Delta(
        claim_id=op.claim_id,
        before_ell=before,
        after_ell=claim.ell,
        cause=cause,
        op=op.op,
    )


def _stable_id(prefix: str, payload: object, event_cursor: int | None = None) -> str:
    blob = {"payload": payload, "event_cursor": event_cursor}
    digest = hashlib.sha256(
        json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def apply(kb: KB, op: Op, evidence: Evidence | None) -> list[Delta]:
    """Dispatch a single validated op to its handler. APPLY_EVIDENCE moves belief;
    ADD_CLAIM/ADD_EDGE/INVALIDATE_EDGE mutate structure; FLAG_OOD/REJECT record
    without moving belief. Returns the Deltas produced (possibly empty)."""
    if isinstance(op, ApplyEvidence):
        if evidence is None:
            raise ValueError("APPLY_EVIDENCE requires evidence")
        return [apply_evidence(kb, op, evidence)]
    if isinstance(op, AddClaim):
        claim_id = _stable_id(
            "claim",
            {"text": op.text.strip(), "ontology_tags": sorted(op.ontology_tags)},
        )
        if claim_id not in kb.claims:
            kb.add_claim(
                Claim(id=claim_id, text=op.text.strip(), ontology_tags=sorted(op.ontology_tags))
            )
        return []
    if isinstance(op, AddEdge):
        edge_id = _stable_id(
            "edge",
            {"src": op.src, "dst": op.dst, "type": op.type.value},
            kb.event_cursor,
        )
        if edge_id not in kb.edges:
            kb.add_edge(
                Edge(
                    id=edge_id,
                    src=op.src,
                    dst=op.dst,
                    type=op.type,
                    weight=op.weight,
                    valid_at=kb.event_cursor,
                )
            )
        return []
    if isinstance(op, InvalidateEdge):
        kb.invalidate_edge(op.edge_id)
        return []
    return []
