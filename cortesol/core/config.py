"""Tunable constants — the single source of truth for every magic number.

FROZEN CONTRACT (steward: Branch 1). No branch may hard-code any of these values
locally; import them from here. They are "tune on the simulator" defaults, but
changing one is a *contract amendment* (bump CONTRACT_VERSION, note it in
docs/CONTRACTS.md changelog, tell the other two branches) because the engine,
the simulator's gold ops, and the eval baselines must all agree on them.

Symbols map to Research/"Confidence Math". δ_max, the strength map, ρ, and the
priors are the knobs Possible Directions §open-questions flags for tuning.
"""

from __future__ import annotations

# Bump this whenever any FROZEN CONTRACT file changes. Every snapshot and eval
# run records it so results are comparable only within a version.
CONTRACT_VERSION = "0.2.0"

# --- Belief prior ---------------------------------------------------------
# A splashy new finding does NOT start at the confidence its abstract asserts.
# Empirically justified skepticism: P(replicates) ~ 0.4-0.6 before any checks
# (Fraud and Hype Signals §base-rates). ADD_CLAIM starts here with high u.
PRIOR_C_0 = 0.45  # -> ell_0 = logit(0.45) ~= -0.20

# --- Evidence strength -> log-likelihood-ratio (pre-cap) ------------------
# Maps the extractor's coarse strength label to a log-Lambda increment.
STRENGTH_TO_LOGLR: dict[str, float] = {
    "weak": 0.5,
    "moderate": 1.5,
    "strong": 3.0,
}

# --- The validator bound (the architectural guarantee) --------------------
# Max |Δell| a single event may move a single claim, AFTER source cap and n_eff
# damping. This is the blast-radius bound that holds even under total LLM
# compromise (Prompt Injection Defense §Layer-4). Prime Directive PD2.
DELTA_MAX = 2.0

# --- Source reliability ---------------------------------------------------
# A source's report is capped at |log Lambda_R| <= log(tau / phi). These are the
# default priors by venue tier; Dawid-Skene can learn them live (stretch, cut
# line #3). tau = true-report rate, phi = false-report rate.
SOURCE_PRIORS: dict[str, tuple[float, float]] = {
    # tier:            (tau,  phi)   -> cap = log(tau/phi)
    "top_journal": (0.95, 0.05),  # cap ~= 2.94
    "reputable": (0.90, 0.10),  # cap ~= 2.20
    "preprint": (0.75, 0.25),  # cap ~= 1.10
    "weak": (0.60, 0.40),  # cap ~= 0.41  (a sensational weak source barely moves belief)
    "predatory": (0.55, 0.45),  # cap ~= 0.20
    "unknown": (0.70, 0.30),  # cap ~= 0.85
}
DEFAULT_SOURCE_TIER = "unknown"

# --- Correlation damping (double-counting defense) ------------------------
# Within-group correlation for the Kish n_eff discount. Group = lab x method x
# dataset. Independent-lab replications land in a fresh group -> full weight.
RHO_WITHIN_GROUP = 0.6

# --- Subjective logic (the "I don't know" view) ---------------------------
SL_PRIOR_WEIGHT_W = 2  # u = W / (r + s + W); a fresh claim has u = 1
SL_BASE_RATE_A = 0.5  # projected P = b + a*u

# --- Red-flag -> false-report-rate (phi) bumps ----------------------------
# Deterministic screen (Fraud and Hype Signals §battery). Each flag raises the
# evidence's effective phi (or damps Λ), so flagged reports self-discount.
# Applied as: phi_eff = min(PHI_CEILING, phi + sum(bumps)).
RED_FLAG_PHI_BUMP: dict[str, float] = {
    "grim_fail": 0.25,  # impossible mean given integer scale + n
    "statcheck_fail": 0.15,  # reported p != recomputed p
    "p_hacking": 0.05,  # p in [.045, .05) — weight lightly (contested)
    "underpowered": 0.15,  # implausible effect size at tiny n
    "no_prereg": 0.05,
    "predatory_venue": 0.20,
    "discredited_source": 0.39,
}
PHI_CEILING = 0.49  # phi < tau always, so the cap stays positive

# --- Out-of-distribution / conflict quarantine ----------------------------
OOD_SIGMA_THRESHOLD = 3.0  # |theta_new - mu| / sqrt(tau^2 + sigma^2) > 3 -> FLAG_OOD
CONFLICT_MASS_THRESHOLD = 0.5  # Dempster-Shafer K above this -> quarantine, don't fuse

# --- Propagation (TruthFinder + typed belief propagation) -----------------
# imp(edge_type) in [-1, 1] for TruthFinder influence and BP compatibilities.
EDGE_INFLUENCE: dict[str, float] = {
    "replicates": 0.9,  # strongest positive coupling
    "supports": 0.6,
    "depends_on": 0.4,  # asymmetric: dst depends on src
    "contradicts": -0.8,  # heterophilic
}
TRUTHFINDER_GAMMA = 0.3  # logistic damping
TRUTHFINDER_RHO = 0.5  # claim<->claim influence weight
BP_DAMPING = 0.5  # m_new = lambda*m_bp + (1-lambda)*m_old
BP_MAX_ITERS = 50
PROPAGATE_KHOP = 2  # re-propagate only the dirty k-hop neighborhood per event

# --- Retrieval / context budget -------------------------------------------
# The whole extractor prompt must fit the Flash dense-model context. Compact
# state serialization is mandatory, not optional (System Architecture §lifecycle).
CONTEXT_TOKEN_BUDGET = 8192
RETRIEVE_TOP_K = 4  # top-k claims + their 1-hop neighborhood

# --- Per-source rate limits (validator) -----------------------------------
MAX_OPS_PER_SOURCE_PER_EVENT = 3

# --- Untrusted-text control-language screen ------------------------------
# These substrings are intentionally conservative. A hit does not make a claim
# false; it only prevents the event from authorizing a state-changing op.
INJECTION_MARKERS: tuple[str, ...] = (
    "ignore prior instructions",
    "ignore the ledger",
    "set confidence",
    "system:",
    "cortex admin",
    "trigger omega",
    "reconsider repeatedly",
    "approve every operation",
)

OUT_OF_SCOPE_MARKERS: tuple[str, ...] = (
    "small-molecule",
    "small molecule",
    "antibody therapy",
    "gene therapy",
    "market claim",
    "regulatory claim",
)
