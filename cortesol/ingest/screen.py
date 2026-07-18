"""Deterministic red-flag screen — skepticism with receipts (area A).

Reference: Research/"Fraud and Hype Signals" §battery + core/domain.py peptide flags.
Runs REGARDLESS of the LLM. GRIM, statcheck, p-hacking, underpowered, no-prereg,
predatory venue, plus peptide-specific checks (implausible Kd, no control peptide,
purity, single replicate). Each flag raises the evidence's effective phi so
flagged reports self-discount in the engine.
"""

from __future__ import annotations

from ..core.kb import KB
from ..core.schema import Evidence


def screen(evidence: Evidence, kb: KB) -> list[str]:
    """Return the red-flag keys that fire for this evidence (RED_FLAG_PHI_BUMP +
    PEPTIDE_RED_FLAG_PHI_BUMP). Attach them to evidence.red_flags."""
    ...  # TODO
