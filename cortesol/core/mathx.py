"""Pure math helpers shared by the whole system.

FROZEN CONTRACT (steward: Branch 1). No LLM, no I/O, no randomness — every
function here is a deterministic pure function. Belief lives in log-odds; these
are the only sanctioned conversions between log-odds, probability, and evidence
strength. Do not re-derive these anywhere else.

See docs/ONTOLOGY.md §Belief and Research/"Confidence Math".
"""

from __future__ import annotations

import math

__all__ = ["sigmoid", "logit", "clip", "kish_neff", "neff_marginal_factor"]


def sigmoid(ell: float) -> float:
    """Log-odds -> probability. c = 1 / (1 + e^-ell)."""
    if ell >= 0:
        z = math.exp(-ell)
        return 1.0 / (1.0 + z)
    z = math.exp(ell)  # numerically stable for very negative ell
    return z / (1.0 + z)


def logit(c: float) -> float:
    """Probability -> log-odds. Clamps away from {0, 1} to keep ell finite."""
    eps = 1e-9
    c = min(1.0 - eps, max(eps, c))
    return math.log(c / (1.0 - c))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def kish_neff(n: int, rho: float) -> float:
    """Kish effective sample size for n correlated observations.

    n_eff = n / (1 + (n - 1) * rho).  rho=0 -> n (independent); rho=1 -> 1
    (perfectly redundant). Used to discount echoes within a correlation group
    (lab x method x dataset). See Confidence Math §3.
    """
    if n <= 0:
        return 0.0
    return n / (1.0 + (n - 1) * rho)


def neff_marginal_factor(k_index: int, rho: float) -> float:
    """Weight of the k-th (0-based) member of a correlation group.

    The 1st member of a group contributes its full log-likelihood-ratio; each
    later member contributes only the *marginal* n_eff increment it adds. This
    is the multiplier applied to that member's log-Lambda so ten papers from one
    lab don't compound like ten independent replications.
    """
    if k_index <= 0:
        return 1.0
    prev = kish_neff(k_index, rho)
    cur = kish_neff(k_index + 1, rho)
    return max(0.0, cur - prev)
