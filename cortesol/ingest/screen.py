"""Deterministic red-flag screen — skepticism with receipts (area A).

Reference: Research/"Fraud and Hype Signals" §battery + core/domain.py peptide flags.
Runs REGARDLESS of the LLM. GRIM, statcheck, p-hacking, underpowered, no-prereg,
predatory venue, plus peptide-specific checks (implausible Kd, no control peptide,
purity, single replicate). Each flag raises the evidence's effective phi so
flagged reports self-discount in the engine.
"""

from __future__ import annotations

from ..core import config, domain
from ..core.kb import KB
from ..core.schema import Evidence


def screen(evidence: Evidence, kb: KB) -> list[str]:
    """Return the red-flag keys that fire for this evidence (RED_FLAG_PHI_BUMP +
    PEPTIDE_RED_FLAG_PHI_BUMP). Attach them to evidence.red_flags."""
    fields = evidence.fields
    flags: set[str] = set()
    lowered_text = evidence.raw_text.lower()
    if any(marker in lowered_text for marker in config.INJECTION_MARKERS):
        flags.add("prompt_injection")
    if any(marker in lowered_text for marker in config.OUT_OF_SCOPE_MARKERS):
        flags.add("out_of_scope")
    n = fields.get("n")
    p = fields.get("p")
    if p is not None and 0.045 <= float(p) < 0.05:
        flags.add("p_hacking")
    reported_p = fields.get("reported_p")
    recomputed_p = fields.get("recomputed_p")
    if reported_p is not None and recomputed_p is not None:
        if abs(float(reported_p) - float(recomputed_p)) > 0.01:
            flags.add("statcheck_fail")
    if fields.get("prereg") is False:
        flags.add("no_prereg")
    if isinstance(n, (int, float)) and n < 10:
        effect = fields.get("effect_size", fields.get("value"))
        if fields.get("units") == "percent" and effect is not None and abs(float(effect)) >= 50:
            flags.add("underpowered")
    if fields.get("source_tier") == "predatory":
        flags.add("predatory_venue")

    rating = fields.get("group_rating")
    if rating is not None and isinstance(n, (int, float)) and n > 0:
        product = float(rating) * int(n)
        if abs(product - round(product)) > 0.05:
            flags.add("grim_fail")

    metric = str(fields.get("metric", ""))
    units = str(fields.get("units", ""))
    value = fields.get("value")
    if metric in {"Kd", "Ki", "IC50", "EC50"} and value is not None:
        factor_to_pm = {"pM": 1.0, "nM": 1000.0, "uM": 1_000_000.0}.get(units)
        if factor_to_pm and float(value) * factor_to_pm < domain.MIN_PLAUSIBLE_KD_PM:
            flags.add("affinity_below_diffusion_limit")
    if fields.get("control_peptide") is False:
        flags.add("no_control_peptide")
    if "purity_pct" not in fields:
        flags.add("purity_not_reported")
    elif float(fields["purity_pct"]) < domain.MIN_ACCEPTABLE_PURITY_PCT:
        flags.add("low_purity")
    if n == 1:
        flags.add("single_replicate")
    source = kb.get_source(evidence.source_id)
    if source is not None and source.discredited:
        flags.add("discredited_source")

    evidence.red_flags = sorted(flags)
    return evidence.red_flags
