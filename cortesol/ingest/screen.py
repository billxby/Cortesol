"""Deterministic red-flag screen — skepticism with receipts (area A).

Reference: Research/"Fraud and Hype Signals" §battery + core/domain.py peptide flags.
Runs REGARDLESS of the LLM. GRIM, statcheck, p-hacking, underpowered, no-prereg,
predatory venue, plus peptide-specific checks (implausible Kd, no control peptide,
purity, single replicate). Each flag raises the evidence's effective phi so
flagged reports self-discount in the engine.
"""

from __future__ import annotations

from ..core.domain import (
    MIN_ACCEPTABLE_PURITY_PCT,
    MIN_PLAUSIBLE_KD_PM,
)
from ..core.kb import KB
from ..core.schema import Evidence


_BINDING_METRICS = frozenset({"Kd", "Ki", "IC50", "EC50"})
_BINDING_ASSAYS = frozenset({"SPR", "ITC", "BLI", "FP", "binding"})


def _is_binding_assay(fields: dict) -> bool:
    """A binding/affinity measurement (vs. a clinical or in-vivo efficacy claim).
    Purity / control-peptide expectations apply only to these."""
    return fields.get("metric") in _BINDING_METRICS or fields.get("assay") in _BINDING_ASSAYS


def _grim_fails(mean: float, n: int, scale_max: int) -> bool:
    """GRIM: a mean of n integer ratings on a 1..scale_max scale must equal some
    integer total / n. If no integer total rounds to the reported mean, it's
    impossible as stated."""
    if n <= 0:
        return False
    total = round(mean * n)
    # allow the reported mean to be a rounding of total/n to its own precision
    return abs(total / n - mean) > 0.5 / n and abs((total) / n - mean) > 1e-6


def screen(evidence: Evidence, kb: KB) -> list[str]:
    """Return the red-flag keys that fire for this evidence (RED_FLAG_PHI_BUMP +
    PEPTIDE_RED_FLAG_PHI_BUMP). Attach them to evidence.red_flags."""
    f = evidence.fields
    flags: list[str] = []

    # --- generic methodological flags ---
    n = f.get("n")
    p = f.get("p")
    if isinstance(p, (int, float)) and 0.045 <= p < 0.05:
        flags.append("p_hacking")
    if f.get("prereg") is False:
        flags.append("no_prereg")
    if isinstance(n, int) and n == 1:
        flags.append("single_replicate")
    # implausibly large in-vivo effect on a handful of animals
    effect = f.get("value") if f.get("metric") == "percent_inhibition" else None
    if isinstance(effect, (int, float)) and isinstance(n, int) and effect >= 60.0 and n <= 8:
        flags.append("underpowered")

    # GRIM on any reported integer-scale group rating
    rating = f.get("group_rating")
    scale = f.get("rating_scale_max")
    if isinstance(rating, (int, float)) and isinstance(scale, int) and isinstance(n, int):
        if _grim_fails(float(rating), n, scale):
            flags.append("grim_fail")

    # --- venue flag from the seeded Source tier ---
    src = kb.get_source(evidence.source_id)
    if src is not None and src.tier == "predatory":
        flags.append("predatory_venue")

    # --- peptide-specific physical-plausibility flags ---
    if f.get("metric") == "Kd" and f.get("units") == "nM":
        value = f.get("value")
        if isinstance(value, (int, float)) and value * 1000.0 < MIN_PLAUSIBLE_KD_PM:
            # value is in nM; * 1000 -> pM. Below ~1 pM is past the diffusion limit.
            flags.append("affinity_below_diffusion_limit")

    if f.get("control_peptide") is False:
        flags.append("no_control_peptide")

    purity = f.get("purity_pct")
    if isinstance(purity, (int, float)):
        if purity < MIN_ACCEPTABLE_PURITY_PCT:
            flags.append("low_purity")
    elif _is_binding_assay(f):
        # Purity is only expected for a binding assay — don't penalise a clinical
        # efficacy paper for omitting a peptide-synthesis field it never has.
        flags.append("purity_not_reported")

    # --- clinical-evidence flags (real papers; parsed from abstract prose) ---
    # Fire only on EXPLICIT weakness signals — a strong RCT / meta-analysis trips
    # none of these and keeps full weight; a narrative review that merely omits
    # "placebo" is not penalised.
    if f.get("study_type") == "clinical":
        if f.get("uncontrolled_design") and not f.get("controlled"):
            flags.append("uncontrolled")  # open-label / single-arm / observational
        elif f.get("open_label") is True:
            flags.append("unblinded")  # controlled but open-label
        if isinstance(n, int) and n < 10:
            flags.append("underpowered")  # small trial for an efficacy claim
    if f.get("case_report") is True:
        flags.append("case_report")

    evidence.red_flags = flags
    return flags
