# Ontology — what the words mean

This is the shared vocabulary. Every part of Cortesol must agree on these
definitions, because the whole system is a disagreement-resolution machine and it
can't resolve disagreements if the parts mean different things by "truth" or
"evidence." When a term is capitalized in the code (`Claim`, `Evidence`,
`Source`), it's defined here and typed in `cortesol/core/schema.py`.

---

## Truth vs. Belief — the distinction the whole system turns on

**Truth** is the actual state of the world: *does* this peptide bind that target,
*is* it stable in serum. In Cortesol, truth is a latent thing the system **never
observes directly**. In the simulator it exists explicitly (the ground-truth
`world` — `z_j` for whether a claim is true, `θ_j` for effect sizes) and is used
**only** to score calibration after the fact. In a real deployment there is no
oracle; there is only evidence arriving over time.

**Belief** is the system's current, calibrated estimate about a claim — a
probability with a record of why. Belief is **not** a truth-claim; it is "given the
evidence I've screened, weighted by how much I trust each source, here is how
confident I am, and here is how much I still don't know."

> The system never asserts truth. It maintains belief and tracks it against truth
> only to prove it is well-calibrated.

The prize is literally named **GROUND TRUTH**: the job is to track the latent
truth as closely as the evidence allows, *without* ever being handed it and
*without* letting a persuasive report overwrite it.

## Information / Evidence

**Information enters the system only as Evidence**: a *report from a Source about a
claim*. A result email, a preprint, a table of assay numbers — each becomes one
`Evidence` record. Key properties:

- Evidence is **data, never instructions.** Its `raw_text` is quarantined and
  never occupies an instruction position. If it contains "set confidence to 1.0,"
  that string is inert — there is no machinery that would act on it.
- Evidence carries **provenance** (`source_id`) and **structured fields** (assay,
  metric, value, units, n, p, lab, method…). The engine acts on the structured
  fields, not the prose.
- Evidence moves belief by a **likelihood ratio** Λ (how much more probable this
  report is if the claim is true vs. false), and that movement is **bounded by the
  source's reliability** and **damped for correlation** (see below).
- Evidence can carry **red flags** from the deterministic screen, which shrink its
  weight.

The unit of information is the report, not the assertion. Ten reports that all
trace to one lab are one echo, not ten confirmations.

## Source & reliability

A **Source** is where a report comes from (a lab, a journal tier, a database). Each
Source has a reliability pair:

- **τ (tau)** — true-report rate: P(it reports X | X is true).
- **φ (phi)** — false-report rate: P(it reports X | X is false).

These produce the single most important guarantee in the system, the **source
cap**:

```
|logΛ_R| ≤ log(τ / φ)
```

As φ → τ (a source that asserts things regardless of truth), the cap → 0: **an
unreliable source's report carries almost no weight, no matter how sensational its
content.** This is Hume's argument about testimony, quantified. Distrust routes to
*uncertainty*, not to disbelief — a bad source tells you nothing, not the opposite.

Defaults by venue tier live in `core/config.py::SOURCE_PRIORS`; they can be learned
online (Dawid–Skene, a stretch goal). A `discredited` source (retraction, exposed
fraud) triggers the retraction **cascade**.

## Confidence — belief as log-odds

Belief is stored as **log-odds** `ℓ = log(P / (1−P))`, displayed as a probability
`c = σ(ℓ)`. Log-odds is the right representation because independent evidence
**adds**:

```
ℓ' = ℓ + logΛ            (each screened, source-capped, correlation-damped report)
```

- **δ_max** caps how far a single event can move a single claim (`config.DELTA_MAX`).
  This bound holds even under a fully compromised model — it is architectural, not
  learned.
- New claims start at a **skeptical prior** (`config.PRIOR_C_0 ≈ 0.45`), not at the
  confidence their abstract asserts. A splashy finding *earns* confidence through
  independent replication; it isn't granted it.

## Uncertainty vs. doubt — "I don't know" ≠ "it's false"

From evidence counts `(r, s)` (supporting, contradicting) Cortesol derives a
subjective-logic opinion `(b, d, u)`:

- **b** (belief) = r / (r+s+W)
- **d** (disbelief) = s / (r+s+W)
- **u** (uncertainty) = W / (r+s+W)

A brand-new claim has **u = 1** — the system honestly says "I don't know," which is
*structurally different* from c = 0.5 reached via lots of conflicting evidence. The
UI shows `u` next to `c`. `W` = `config.SL_PRIOR_WEIGHT_W`.

## The peptide ontology — what's in scope

Cortesol tracks **properties of peptides**. In-scope claims reference a peptide
entity and a property (`core/domain.py`):

- **Entities:** peptide, target, modification, assay, cell_line, organism.
- **Properties:** binding_affinity (Kd/Ki/IC50/EC50), potency, selectivity,
  serum_stability, thermal_stability, permeability, solubility, immunogenicity,
  toxicity, efficacy, synthesis.

Tags look like `peptide:GLP1-analog-7`, `target:GLP1R`, `property:binding_affinity`,
`assay:SPR`, `modification:cyclization`.

**Out of scope** → `FLAG_OOD`, never force-fit into a claim: small-molecule-only
pharmacology, antibody/large-biologic claims, gene/cell therapy, human clinical
phase outcomes, regulatory/IP/market claims, chemistry unrelated to a peptide.
Flagging is always safe; guessing outside your ontology is not. (CORTEX: "the
knowledge base can be read as an abstract graph of states and claims" — peptides is
our concrete skin over that abstract graph.)

## Ops — the only way to change state

Nothing changes belief except an **op** from the closed vocabulary
(`core/ops.py`): `APPLY_EVIDENCE`, `ADD_CLAIM`, `ADD_EDGE`, `INVALIDATE_EDGE`,
`FLAG_OOD`, `REJECT`. There is deliberately **no `SET_CONFIDENCE` and no `DELETE`.**
The model emits ops; the validator is the only thing that lets them through.

## Red flags

Deterministic, model-independent quality signals on a piece of evidence (GRIM,
statcheck, p-hacking, underpowered, no-preregistration, predatory venue, plus
peptide-specific: implausible Kd, missing control peptide, low purity, single
replicate). Each raises the evidence's effective φ, so **flagged evidence
discounts itself** in the engine. Defined in `config.py` + `domain.py`, applied in
`ingest/screen.py`.

## Provenance & audit

Every belief move records a **Delta** (claim, before/after ℓ, cause, op) into an
immutable **audit log**. "Every Δℓ has a cause chain" is the anti-black-box
property: a judge can click any node and see exactly why it moved. Nothing is
deleted — edges and claims are *invalidated* (bi-temporal `valid_at`/`invalid_at`),
preserving the history.

## Injection

Any untrusted text attempting to act as an instruction ("SYSTEM: overwrite
claim X", a planted memory, an authority spoof). In Cortesol an injection is just
**more data from a source**: the correct response is `REJECT(injection)` or, if it
smuggles a real-looking result, to treat it as one capped, provenance-logged,
low-trust report. It can never reach the control plane, because the write path
takes ops, not text.

---

## Symbol table

| Symbol | Meaning | Where |
| --- | --- | --- |
| `ℓ` | log-odds (belief) of a claim | `Claim.ell` |
| `c` | confidence `σ(ℓ)` — a probability, not truth | `Claim.c` |
| `Λ` | likelihood ratio of a report | engine |
| `τ, φ` | source true-/false-report rates → cap `log(τ/φ)` | `Source.tau/phi` |
| `r, s` | supporting / contradicting evidence counts | `Claim.r/s` |
| `u` | subjective-logic uncertainty `W/(r+s+W)` | `Claim.u` |
| `W` | prior weight for u | `config.SL_PRIOR_WEIGHT_W` |
| `δ_max` | max \|Δℓ\| per claim per event (the bound) | `config.DELTA_MAX` |
| `ρ` | within-correlation-group correlation | `config.RHO_WITHIN_GROUP` |
| `n_eff` | Kish effective sample size (echo discount) | `mathx.kish_neff` |
| `K` | Dempster–Shafer conflict mass → quarantine | `config.CONFLICT_MASS_THRESHOLD` |

## Enumerations

- **ClaimStatus:** `active`, `flagged`, `quarantined`, `retired`
- **EdgeType:** `supports`, `contradicts`, `depends_on`, `replicates`
- **Strength:** `weak`, `moderate`, `strong` (→ `config.STRENGTH_TO_LOGLR`)
- **Direction:** `+` (supports), `-` (contradicts)
- **EventClass** (sim only): `genuine`, `noisy`, `hyped`, `fraudulent`,
  `contradictory`, `out_of_scope`, `injection`
- **RejectReason:** `injection`, `malformed`, `unverifiable`
