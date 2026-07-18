"""Deterministic structured-field extraction from free-text abstracts (area A).

Real papers arrive as prose (title + abstract), not the structured `fields` the
simulator emits. This module parses the numbers and study-design signals the
deterministic screen needs — affinity/effect metrics, sample size, p-value, and
the clinical-trial design (randomized / blinded / placebo-controlled / meta-
analysis / case report). It is pure, dependency-free, and reads the untrusted text
only as DATA (regex over prose — nothing is executed).

The extractor and screen consume the result; belief still moves only through the
engine. `parse_fields` is called by `quarantine` ONLY when an event has no
structured metric (i.e. real papers), so the simulator path is untouched.

Reference: Fraud and Hype Signals §battery (the fields those checks read).
"""

from __future__ import annotations

import re
from typing import Any

# A signed number, allowing thousands separators and a unicode minus.
_NUM = r"[-−]?\d[\d,]*(?:\.\d+)?"

# Binding affinity: requires "<metric> <op> <number> <unit>" so bare letters like
# "Ki" inside words ("taking") never match. Units normalised to nM downstream.
_AFFINITY_RE = re.compile(
    r"\b(Kd|Ki|IC50|EC50)\b\s*(?:value\s*)?(?:of|=|~|≈|:|was|is|of about)?\s*"
    r"(" + _NUM + r")\s*(pM|nM|µM|uM|μM|mM)",
    re.IGNORECASE,
)
_N_RE = re.compile(r"\b[nN]\s*=\s*(\d[\d,]*)")
_COHORT_RE = re.compile(
    r"\b(\d[\d,]*)\s+(?:patients|participants|individuals|subjects|adults|"
    r"men|women|mice|rats|animals)\b",
    re.IGNORECASE,
)
_P_RE = re.compile(r"\b[pP]\s*[<=≤]\s*(0?\.\d+)")
_EFFECT_RE = re.compile(
    r"(?:reduc\w+|decreas\w+|loss|lower\w*|increas\w+|improv\w+|weight loss)"
    r"[^.]{0,40}?(" + _NUM + r")\s*%"
    r"|(" + _NUM + r")\s*%[^.]{0,25}?(?:reduction|decrease|loss|lower|weight)",
    re.IGNORECASE,
)

_UNIT_TO_NM = {"pm": 0.001, "nm": 1.0, "µm": 1000.0, "um": 1000.0, "μm": 1000.0, "mm": 1_000_000.0}

_DESIGN = {
    "randomized": re.compile(r"randomi[sz]ed|\brct\b", re.I),
    "blinded": re.compile(r"double[-\s]?blind|single[-\s]?blind|\bblinded\b", re.I),
    "controlled": re.compile(r"placebo|controlled|comparator|versus placebo|vs\.? placebo", re.I),
    "meta_analysis": re.compile(r"meta[-\s]?analysis|systematic review|pooled analysis", re.I),
    "case_report": re.compile(r"case report|case series|a case of|we report a", re.I),
    "in_vitro": re.compile(r"\bin vitro\b|cell[-\s]?based assay|binding assay", re.I),
    # explicit weakness markers — only these justify an "uncontrolled" flag, so a
    # narrative review that merely omits the word "placebo" is NOT penalised.
    "open_label": re.compile(r"open[-\s]?label", re.I),
    "uncontrolled_design": re.compile(
        r"open[-\s]?label|single[-\s]?arm|observational|retrospective|prospective cohort|"
        r"non[-\s]?randomized|non[-\s]?randomised|uncontrolled|case series|real[-\s]?world",
        re.I,
    ),
}


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def parse_fields(text: str) -> dict[str, Any]:
    """Extract structured fields from abstract prose. Returns only the keys it can
    find (plus explicit design booleans when the paper is clinical, so the screen
    can distinguish 'uncontrolled' from 'unknown')."""
    out: dict[str, Any] = {}
    if not text:
        return out
    low = text.lower()

    # --- affinity (rare in a clinical corpus, but handled for generality) ---
    m = _AFFINITY_RE.search(text)
    if m:
        val = _to_float(m.group(2))
        unit = m.group(3).lower()
        if val is not None:
            out["metric"] = m.group(1).upper() if m.group(1).upper() != "KD" else "Kd"
            out["value"] = round(val * _UNIT_TO_NM.get(unit, 1.0), 6)
            out["units"] = "nM"
            out["assay"] = "binding"

    # --- sample size: largest of n= / N= / cohort phrases ---
    sizes = [int(x.replace(",", "")) for x in _N_RE.findall(text)]
    sizes += [int(x.replace(",", "")) for x in _COHORT_RE.findall(text)]
    if sizes:
        out["n"] = max(sizes)

    # --- p-value: the smallest (most significant) reported ---
    ps = [p for p in (_to_float(x) for x in _P_RE.findall(text)) if p is not None]
    if ps:
        out["p"] = min(ps)

    # --- design signals ---
    design = {k: bool(rx.search(text)) for k, rx in _DESIGN.items()}

    # --- clinical effect size (percent), if no binding metric ---
    if "metric" not in out:
        em = _EFFECT_RE.search(text)
        if em:
            eff = _to_float(em.group(1) or em.group(2) or "")
            if eff is not None:
                out["metric"] = "percent_change"
                out["value"] = eff
                out["units"] = "percent"

    # --- study type + explicit design booleans for the screen ---
    is_clinical = bool(
        design["randomized"] or design["controlled"] or design["case_report"]
        or out.get("metric") == "percent_change"
        or re.search(r"patients|participants|trial|weight loss|efficacy", low)
    )
    if out.get("assay") == "binding":
        out["study_type"] = "binding"
    elif design["meta_analysis"]:
        out["study_type"] = "review"  # high-quality synthesis — not penalised
    elif is_clinical:
        out["study_type"] = "clinical"
    elif design["in_vitro"]:
        out["study_type"] = "in_vitro"

    # Expose booleans so the screen can flag 'uncontrolled'/'unblinded' vs unknown.
    if out.get("study_type") in ("clinical", "review"):
        out["randomized"] = design["randomized"]
        out["blinded"] = design["blinded"]
        out["controlled"] = design["controlled"]
        if design["open_label"]:
            out["open_label"] = True
        if design["uncontrolled_design"]:
            out["uncontrolled_design"] = True
    if design["case_report"]:
        out["case_report"] = True
    reported = [k for k, v in design.items() if v and k not in ("open_label", "uncontrolled_design")]
    out["study_design"] = reported or ["unspecified"]

    return out
