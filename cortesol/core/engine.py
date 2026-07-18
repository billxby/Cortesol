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

from .kb import KB
from .ops import ApplyEvidence, Op
from .results import Delta
from .schema import Evidence


def strength_to_loglr(strength: str) -> float:
    """Map weak/moderate/strong -> pre-cap log-Lambda (config.STRENGTH_TO_LOGLR)."""
    raise NotImplementedError("BRANCH-1: trivial lookup into config.STRENGTH_TO_LOGLR")


def source_cap(tau: float, phi: float) -> float:
    """|log Lambda_R| <= log(tau/phi). The anti-hype guarantee (Confidence Math §2).
    A weak source (tau~phi) yields a cap near 0: sensational content, tiny move."""
    raise NotImplementedError("BRANCH-1: math.log(tau / phi)")


def fraud_switch_factor(red_flags: list[str], phi: float) -> float:
    """Return a multiplier in [0, 1] that shrinks Lambda as red flags raise the
    effective phi (Confidence Math §2 mixture; bumps in config + domain)."""
    raise NotImplementedError("BRANCH-1: combine RED_FLAG_PHI_BUMP + PEPTIDE_RED_FLAG_PHI_BUMP")


def neff_factor(kb: KB, claim_id: str, evidence: Evidence) -> float:
    """Marginal n_eff weight for this evidence given how many correlated reports
    (same correlation_group) the claim has already absorbed (mathx.neff_marginal_factor).
    First report in a group -> 1.0; echoes -> shrinking. Updates claim.correlation_seen."""
    raise NotImplementedError("BRANCH-1: use claim.correlation_seen + mathx.neff_marginal_factor")


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
    raise NotImplementedError("BRANCH-1: implement the Confidence Math §engine recipe")


def apply(kb: KB, op: Op, evidence: Evidence | None) -> list[Delta]:
    """Dispatch a single validated op to its handler. APPLY_EVIDENCE moves belief;
    ADD_CLAIM/ADD_EDGE/INVALIDATE_EDGE mutate structure; FLAG_OOD/REJECT record
    without moving belief. Returns the Deltas produced (possibly empty)."""
    raise NotImplementedError("BRANCH-1: dispatch on op type; only APPLY_EVIDENCE moves ell")
