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


# --- Curated grounded peptide knowledge -----------------------------------
# The claims Cortesol *seeds* so the graph starts with real structure instead of
# "peptide binds target" alone. Every entry is a well-established property of the
# peptide (grounded in the mainstream literature), expressed in the ontology above.
#
# IMPORTANT — this seeds CLAIM NODES, never belief. New claims are created at the
# skeptical prior (PRIOR_C_0); confidence is still EARNED only when real-paper
# evidence flows through the engine. A seeded claim that never receives evidence
# simply stays hidden (the UI only reveals claims a committed result has touched),
# so this widens the graph without cluttering it.
#
# Each entry: name (lowercase canonical) -> {
#   "target": the primary molecular target for the binding claim, or None when the
#             peptide has no single confirmed receptor (then no "binds X" claim —
#             this is how we avoid ever seeding "binds unknown"),
#   "claims": list of (property_namespace, slug, human_text, keyword_cues) — the
#             extractor routes a paper's evidence to the claim whose cues match the
#             abstract, so real papers move these property claims.
# }
# Widening this table needs NO retraining and NO contract change: ADD_CLAIM is
# free-text over the open property namespaces above; only the op/edge vocabulary is
# the frozen, retraining-implicated part. (docs/ONTOLOGY.md §Truth vs Belief.)

PEPTIDE_KNOWLEDGE: dict[str, dict] = {
    "semaglutide": {
        "target": "GLP1R",
        "claims": [
            ("efficacy", "obesity", "semaglutide reduces body weight in obesity",
             ("obes", "weight", "bmi", "adipos", "overweight")),
            ("efficacy", "type_2_diabetes",
             "semaglutide improves glycemic control in type 2 diabetes",
             ("diabet", "hba1c", "glyc", "glucose", "insulin")),
            ("efficacy", "cardiovascular", "semaglutide reduces major cardiovascular events",
             ("cardiovascular", "mace", "cardiac", "stroke", "heart")),
            ("serum_stability", "long_half_life",
             "semaglutide has a ~1-week half-life enabling once-weekly dosing",
             ("half-life", "half life", "once-weekly", "once weekly", "long-acting")),
        ],
    },
    "tirzepatide": {
        "target": "GLP1R/GIPR",
        "claims": [
            ("efficacy", "obesity", "tirzepatide produces substantial weight loss in obesity",
             ("obes", "weight", "bmi", "adipos", "overweight")),
            ("efficacy", "type_2_diabetes",
             "tirzepatide improves glycemic control in type 2 diabetes",
             ("diabet", "hba1c", "glyc", "glucose", "insulin")),
            ("selectivity", "dual_agonist", "tirzepatide co-activates the GLP-1 and GIP receptors",
             ("gip", "dual", "co-agonist", "twincretin", "unimolecular")),
        ],
    },
    "retatrutide": {
        "target": "GLP1R/GIPR/GCGR",
        "claims": [
            ("efficacy", "obesity", "retatrutide produces large weight loss in obesity",
             ("obes", "weight", "bmi", "adipos")),
            ("selectivity", "triple_agonist",
             "retatrutide is a GLP-1 / GIP / glucagon receptor triple agonist",
             ("triple", "glucagon", "gcgr", "agonist")),
        ],
    },
    "liraglutide": {
        "target": "GLP1R",
        "claims": [
            ("efficacy", "obesity", "liraglutide reduces body weight in obesity",
             ("obes", "weight", "bmi")),
            ("efficacy", "type_2_diabetes",
             "liraglutide improves glycemic control in type 2 diabetes",
             ("diabet", "hba1c", "glyc", "glucose")),
        ],
    },
    "exenatide": {
        "target": "GLP1R",
        "claims": [
            ("efficacy", "type_2_diabetes",
             "exenatide improves glycemic control in type 2 diabetes",
             ("diabet", "hba1c", "glyc", "glucose")),
        ],
    },
    "cagrilintide": {
        "target": "amylin receptor",
        "claims": [
            ("efficacy", "obesity", "cagrilintide reduces body weight in obesity",
             ("obes", "weight", "bmi", "appetite")),
        ],
    },
    "setmelanotide": {
        "target": "MC4R",
        "claims": [
            ("efficacy", "obesity", "setmelanotide reduces weight in MC4R-pathway genetic obesity",
             ("obes", "weight", "hyperphag", "appetite")),
        ],
    },
    "bremelanotide": {
        "target": "MC4R",
        "claims": [
            ("efficacy", "sexual_dysfunction", "bremelanotide improves hypoactive sexual desire",
             ("sexual", "libido", "desire", "hsdd", "arousal")),
        ],
    },
    "melanotan-ii": {
        "target": "MC1R/MC4R",
        "claims": [
            ("efficacy", "melanogenesis",
             "melanotan-II stimulates melanogenesis and skin pigmentation",
             ("melano", "tan", "pigment", "skin")),
        ],
    },
    "teriparatide": {
        "target": "PTH1R",
        "claims": [
            ("efficacy", "osteoporosis",
             "teriparatide increases bone mineral density in osteoporosis",
             ("bone", "osteopor", "fracture", "bmd", "skeletal")),
        ],
    },
    "octreotide": {
        "target": "SSTR2",
        "claims": [
            ("efficacy", "acromegaly", "octreotide suppresses growth hormone in acromegaly",
             ("acromegal", "growth hormone", "igf", " gh ")),
            ("efficacy", "neuroendocrine", "octreotide controls neuroendocrine tumor symptoms",
             ("neuroendocrine", "carcinoid", "tumor", "tumour")),
        ],
    },
    "linaclotide": {
        "target": "guanylate cyclase-C",
        "claims": [
            ("efficacy", "constipation", "linaclotide relieves constipation in IBS-C",
             ("constipat", "ibs", "bowel", "abdominal")),
        ],
    },
    "ziconotide": {
        "target": "N-type calcium channel",
        "claims": [
            ("efficacy", "chronic_pain", "ziconotide relieves severe chronic pain",
             ("pain", "analges", "intrathecal")),
        ],
    },
    "icatibant": {
        "target": "bradykinin B2 receptor",
        "claims": [
            ("efficacy", "angioedema", "icatibant resolves hereditary angioedema attacks",
             ("angioedema", "hae", "swelling", "attack")),
        ],
    },
    "degarelix": {
        "target": "GnRH receptor",
        "claims": [
            ("efficacy", "prostate_cancer", "degarelix suppresses testosterone in prostate cancer",
             ("prostate", "testosterone", "androgen")),
        ],
    },
    "enfuvirtide": {
        "target": "gp41",
        "claims": [
            ("efficacy", "hiv", "enfuvirtide inhibits HIV-1 fusion and entry",
             ("hiv", "fusion", "viral load", "antiretroviral", "entry")),
        ],
    },
    "elamipretide": {
        "target": "cardiolipin",
        "claims": [
            ("efficacy", "mitochondrial", "elamipretide is studied for mitochondrial dysfunction",
             ("mitochondr", "cardiolipin", "muscle")),
        ],
    },
    "bpc-157": {
        "target": None,  # no confirmed receptor — carried by its (preclinical) efficacy claim
        "claims": [
            ("efficacy", "tissue_repair",
             "BPC-157 promotes tendon and tissue healing (preclinical)",
             ("tendon", "heal", "tissue", "repair", "ligament", "injur", "wound")),
        ],
    },
    "thymosin-beta-4": {
        "target": "actin",
        "claims": [
            ("efficacy", "tissue_repair", "thymosin beta-4 promotes tissue repair and angiogenesis",
             ("heal", "repair", "angiogen", "wound", "regener")),
        ],
    },
    "ghk-cu": {
        "target": None,  # a copper-binding tripeptide, not a receptor ligand
        "claims": [
            ("efficacy", "skin", "GHK-Cu improves skin quality and wound healing",
             ("skin", "wound", "dermal", "heal", "wrinkle")),
            ("efficacy", "collagen",
             "GHK-Cu stimulates collagen and extracellular-matrix synthesis",
             ("collagen", "matrix", "fibroblast", "regener")),
        ],
    },
    "selank": {
        "target": None,  # modulates GABA/BDNF systems; no single receptor target
        "claims": [
            ("efficacy", "anxiolytic", "Selank has anxiolytic effects (research)",
             ("anxi", "stress", "anxiolytic", "cognit")),
        ],
    },
}


def binding_claim_id(peptide: str, target: str) -> str:
    """Canonical id for a peptide's binding claim. The ONE place this scheme lives —
    both the KB seeder (adapters/cortex.py) and the offline extractor derive ids from
    here so they always agree."""
    return f"c_bind_{peptide}_{target}"


def property_claim_id(namespace: str, peptide: str, slug: str) -> str:
    """Canonical id for a seeded property claim, e.g. c_efficacy_semaglutide_obesity."""
    return f"c_{namespace}_{peptide}_{slug}"


def peptide_facts(peptide: str | None) -> dict | None:
    """Grounded knowledge for a peptide by (case-insensitive) canonical name, or None."""
    if not peptide:
        return None
    return PEPTIDE_KNOWLEDGE.get(peptide.strip().lower())


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
    replications. (Confidence Math §3.)

    Simulator events carry lab/method/dataset directly. Real papers don't, so we
    fall back to journal/assay: two different journals reporting the same claim are
    then treated as *independent* corroboration (full weight), while repeats from
    the same venue still damp as echoes — otherwise every paper would collapse into
    one 'unknown' bucket and genuine replication would be discounted."""
    lab = fields.get("lab") or fields.get("journal") or "?"
    method = fields.get("method") or fields.get("assay") or "?"
    dataset = fields.get("dataset") or "?"
    return "|".join(str(x) for x in (lab, method, dataset))


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
