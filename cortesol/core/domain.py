"""The peptides-research domain — the ontology the KB is built to represent.

FROZEN CONTRACT (steward: Branch 1, co-authored with Branch 2 who generates data
in this vocabulary). The belief *engine* is domain-agnostic; THIS file is the one
place the peptide subject matter lives in code. It defines: what entities/
properties exist, which claims are in-scope, the structured evidence-field schema
the screen and engine read, and the peptide-specific red flags.

CORTEX says "no science background required; the KB can be read as an abstract
graph of states and claims." Peptides is our concrete instantiation of that
abstract graph — see docs/ONTOLOGY.md §Domain. Anything outside this ontology is
routed to FLAG_OOD, never force-fit into a claim.
"""

from __future__ import annotations

from typing import Any

# --- Ontology: entity and property types (the tag namespace) --------------

ENTITY_TYPES: tuple[str, ...] = (
    "peptide",  # e.g. peptide:GLP1-analog-7, a sequence under study
    "target",  # e.g. target:GLP1R, the protein it acts on
    "modification",  # e.g. modification:cyclization, PEGylation, D-amino-sub
    "assay",  # e.g. assay:SPR
    "cell_line",  # e.g. cell_line:HEK293
    "organism",  # e.g. organism:mouse
)

PROPERTY_TYPES: tuple[str, ...] = (
    "binding_affinity",  # Kd / Ki / IC50 / EC50
    "potency",  # functional EC50 / percent inhibition
    "selectivity",  # on-target vs off-target ratio
    "serum_stability",  # half-life in serum/plasma
    "thermal_stability",  # Tm
    "permeability",  # cell / blood-brain-barrier penetration
    "solubility",  # aqueous solubility / aggregation propensity
    "immunogenicity",
    "toxicity",  # cytotoxicity / in-vivo tox
    "efficacy",  # in-vivo disease-model effect (preclinical)
    "synthesis",  # yield / purity / route
)

# A claim tag looks like "<entity_or_property>:<slug>". In-scope claims must
# reference at least one peptide entity AND one property (Cortesol tracks
# *properties of peptides*, not free-floating facts).
IN_SCOPE_NAMESPACES: frozenset[str] = frozenset(ENTITY_TYPES + PROPERTY_TYPES)

# Explicitly out-of-scope — route to FLAG_OOD, do not coerce into a claim.
# (docs/ONTOLOGY.md §Out-of-scope. The sim's `out_of_scope` class draws from here.)
OUT_OF_SCOPE_EXAMPLES: tuple[str, ...] = (
    "small-molecule pharmacology with no peptide entity",
    "antibody / large-biologic claims (not peptides)",
    "gene or cell therapy",
    "human clinical-trial phase outcomes (KB is preclinical)",
    "regulatory / IP / market claims",
    "generic chemistry unrelated to a peptide or its target",
)


# --- Structured evidence-field schema -------------------------------------
# The keys the deterministic screen (Branch 1) and the engine read out of
# Evidence.fields. The extractor (Branch 2) populates these from raw_text; the
# simulator (Branch 2) emits them directly. Missing keys are allowed — the
# screen treats absence as a (mild) red flag where relevant.

EVIDENCE_FIELD_KEYS: tuple[str, ...] = (
    "assay",  # SPR | ITC | BLI | FP | cell_viability | ELISA | HPLC | MS | in_vivo
    "metric",  # Kd | Ki | IC50 | EC50 | half_life | Tm | percent_inhibition | yield
    "value",  # float
    "units",  # nM | uM | pM | hours | celsius | percent
    "n",  # int, biological replicates
    "p",  # float | null
    "effect_size",  # float | null
    "cell_line",  # str | null
    "organism",  # str | null
    "lab",  # str  — part of the correlation group
    "method",  # str  — part of the correlation group
    "dataset",  # str | null — part of the correlation group
    "prereg",  # bool
    "control_peptide",  # bool — scrambled / negative control present
    "purity_pct",  # float | null
)

ASSAY_TYPES: tuple[str, ...] = (
    "SPR",
    "ITC",
    "BLI",
    "FP",
    "cell_viability",
    "ELISA",
    "HPLC",
    "MS",
    "in_vivo",
)


def correlation_group(fields: dict[str, Any]) -> str:
    """The n_eff bucket for a piece of evidence: lab x method x dataset. Reports
    sharing a bucket are treated as correlated (echoes), not independent
    replications. (Confidence Math §3.)"""
    return "|".join(str(fields.get(k, "?")) for k in ("lab", "method", "dataset"))


# --- Peptide-specific red flags (extend the generic screen) ----------------
# Merged with core.config.RED_FLAG_PHI_BUMP by the screen. Each raises the
# evidence's effective false-report rate phi, so flagged reports self-discount.
# (Fraud and Hype Signals §battery, peptide instantiation.)

PEPTIDE_RED_FLAG_PHI_BUMP: dict[str, float] = {
    "affinity_below_diffusion_limit": 0.30,  # Kd < ~1 pM is physically implausible
    "no_control_peptide": 0.15,  # no scrambled / negative control
    "purity_not_reported": 0.10,
    "low_purity": 0.15,  # crude peptide (<95%) claimed as pure result
    "single_replicate": 0.15,  # n == 1
    "aggregation_ignored": 0.05,  # solubility claimed for aggregation-prone seq
    # --- clinical-evidence flags (real papers; parsed from abstract prose) ---
    # Additive extension of the screen for clinical efficacy claims. A strong RCT /
    # meta-analysis fires none of these (full weight); a weak observational report
    # or case study fires several and self-discounts. (Fraud and Hype Signals.)
    "uncontrolled": 0.15,  # no placebo / comparator arm
    "unblinded": 0.10,  # open-label where blinding was feasible
    "case_report": 0.12,  # anecdotal (case report / series)
}

# Physical sanity bounds for the deterministic screen.
MIN_PLAUSIBLE_KD_PM = 1.0  # picomolar; tighter than this for a peptide is suspect
MIN_ACCEPTABLE_PURITY_PCT = 95.0
