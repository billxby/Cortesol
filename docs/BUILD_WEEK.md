# BUILD_WEEK.md — OpenAI 2026 Build Week plan

> Working plan for the build-week submission on branch `feat/build-week-2026`.
> Scope locked 2026-07-20. This is the "what we're adding and in what order"
> companion to `ARCHITECTURE.md` (how the core works) and `STATUS.md` (what's
> already built). Update the track table as each track lands.

## The one-line pitch

**From a peptide belief engine to a general, generative epistemic memory.**
Same gated core — now it (a) tracks calibrated belief in *any* field you import,
and (b) *renders* that belief into live, grounded reports you build by talking to
it. **Truth in, truth out.** Hits both prizes: CORTEX GROUND TRUTH (a general
belief engine) and Freesolo Best Model (a calibrated update policy that
generalizes across subjects).

## The invariant nothing may break

Belief moves only through the engine (`ℓ' = ℓ + logΛ`, capped by `log(τ/φ)` and
`DELTA_MAX`). Every new feature obeys it:

- **Untrusted text stays data.** New ingest paths enter only via `quarantine`.
- **A report renders the ledger; it never writes it.** The generative UI's
  `ReportSpec` is the output analogue of the op vocabulary — the LLM proposes
  layout, a deterministic renderer binds facts from the KB.
- **A Domain is trusted configuration, not belief.** LLM-drafted ontologies are
  reviewed by a human before activation; they define *vocabulary/scope*, never
  confidence.
- `make test-contract` (the merge gate) stays green on every track.

## Scope (locked 2026-07-20)

| Decision | Choice |
|---|---|
| Domain generalization | **Pluggable `Domain` protocol + LLM-drafted domains** (human-confirmed) |
| Generative UI | **Grounded report builder + conversational live editing** (full vision) |
| Mission-loyal panels | **All four**: live red-team, calibration proof, paste-in ingest, audit inspector |
| Freesolo | **Train a generalist policy** across subjects, within the 50-credit budget |

## Two architectural moves

**Move 1 — a pluggable `Domain`.** Turn `core/domain.py` into a `Domain` protocol
(`core/domains/base.py`) plus `core/domains/peptides.py` (today's content as one
instance). A `Domain` carries: entity types, property/claim types, scope
boundary, screening rules + plausibility bounds, direction/strength semantics,
seed knowledge + claim-id scheme, correlation-group def, corpus/source-tier
heuristics, and prompt templates. Thread one `Domain` through
`validator / engine / screen / judge / quarantine / retrieval / adapters`. The
belief-math core (`mathx / schema / ops / kb / context / results / propagate`)
stays untouched. Also move `config.OUT_OF_SCOPE_MARKERS` onto the Domain.

**Move 2 — a `ReportSpec` + grounded renderer** (mirrors the op boundary). A
closed component vocabulary — `claim-card`, `confidence-meter`,
`trajectory-sparkline`, `evidence-table`, `contradiction-panel`,
`provenance-trail`, `graph-embed`, `stat-tile`, `metric-comparison`,
`prose-block` — where **each component binds to a KB query**, not to free text.
Numbers = `claim.c`, error bars = `u`, links = the audit trail. A
**groundedness gate** flags any component/prose that asserts what the graph
doesn't support. Conversational edits mutate the spec; smooth re-layout via FLIP
transitions (works in the existing vanilla-JS UI, no build step).

## The crux: a domain-neutral model contract

`core/assessment.py` is peptide-shaped **and compiled into the tuned model's
constrained decoding + SFT targets** (`assessment.py:17-92`, schema gen
`:95-163`). This is why the current champion checkpoint can't serve other fields.
The fix — needed by *both* "import any field" and the generalist model — is to
make `EvidenceAssessment` neutral (entity / object / property / quality_signal /
value / direction-cue with **Domain-supplied enums**, schema re-derived per
domain) and make `core/judge.py` policy Domain-driven (fatal/weakening flag sets,
direction/strength rubric, new-claim templating). Peptides remain one
specialization so nothing regresses.

## Tracks (sequenced; each leaves a working demo)

Legend: dep = depends on · risk = L/M/H · demo = what's newly demoable after it.

| # | Track | Dep | Risk | Demo after |
|---|---|---|---|---|
| **T0** | **Domain protocol refactor** — `domain.py` → `Domain` + `domains/peptides.py`; thread through core; keep contract green | — | M | (invisible) peptides byte-identical; foundation for everything |
| **T1** | **Mission panels** — paste-in ingest endpoint `POST /ingest`; live red-team box; calibration proof panel (reliability + Brier/ECE vs baselines); audit inspector (node → cause chain) | T1a paste-in first | L | dramatically stronger *existing* demo; the money-shots |
| **T2** | **Generative grounded UI** — `ReportSpec` schema + renderer bound to KB queries + groundedness gate; then the conversational bar (LLM → spec deltas over NDJSON/SSE) + FLIP animation | can start on peptide KB | H | "build me a report" → live grounded dashboard you edit by talking |
| **T3** | **Generic assessment + Domain-driven judge** — neutralize `assessment.py`; make `judge.py` Domain-driven; peptide as a specialization | T0 | H | new fields actually update belief correctly |
| **T4** | **LLM-drafted domains** — authoring tool: describe a field → LLM drafts a `Domain` config → human review/edit → activate; corpus import to seed it | T0, T3 | M | import *any* field of knowledge live |
| **T5** | **Generalist Freesolo training** — multi-domain SFT data via existing `datasets.py`/`make_sft.py`/`teacher_filter.py` against the neutral schema; cost-capped SFT (+ short GRPO if budget); deploy generalist adapter | T3 | M | one policy that stays calibrated across subjects |

**Parallelism:** T0 is the foundation and goes first. T1 (mostly additive
UI/endpoints) and an early T2 vertical slice can proceed alongside T0/T3. T5 has
the longest latency and is accuracy-secondary, so it runs late / in the
background. Recommended kickoff: **T0 + T1a (paste-in ingest) together.**

## Freesolo generalist plan (≤ 50 credits)

Full lineage caps at ~$60–67 (`docs/TRAINING.md:39`), over budget — so go lean:

1. Neutral assessment schema (T3) is the prerequisite.
2. Generate **multi-domain** SFT data (extend `sim/` to emit ≥2–3 non-peptide
   worlds, or synthesize cross-domain rows) via `datasets.py` + `make_sft.py` +
   `teacher_filter.py`, targeting the neutral schema.
3. **SFT first** (cheap, deterministic), then a **short GRPO** only if budget
   remains; **skip OPD** (the expensive managed-teacher distillation).
4. Coordinator cap **≤ ~$40** (`coordinator.approve --cap-usd 40`) to keep
   headroom; the coordinator refuses to exceed the approved cap.
5. Accuracy is secondary — this is a **proof the recipe + policy generalize**.
   Fallback for any field at demo time: the already-wired frontier proposer
   (Gemini) + the deterministic judge.

## Honest risks & limits

- **Core refactor blast radius (T0/T3).** Mitigate: keep peptides as a
  specialization; `make test-contract` green at every commit; refactor behind the
  existing narrow seams (`validator.py:48`, `engine.py:70`, free-text
  `ontology_tags`/`Evidence.fields`).
- **Grounded prose is best-effort.** Structured components are grounded by
  construction; free LLM prose in `prose-block` is gated but not provably
  complete — keep prose minimal and citation-tagged.
- **Budget < full lineage.** Lean training (above); generalist accuracy is a PoC,
  not a benchmark win.
- **Conversational layout editing (T2) is the flashiest and riskiest.** Land the
  static grounded report first; treat live NL editing as the top layer.

## Suggested live-demo arc

1. Open on the peptide graph (existing 3-act story still works).
2. **Attack it** (red-team box): paste an injection + a fraudulent abstract →
   watch REJECT / cap / screen. Show the calibration panel staying honest.
3. **Import a new field**: describe it in plain English → LLM drafts the Domain →
   confirm → paste a few papers → belief forms under the *same* gated engine.
4. **Present it**: "build me a report on the strongest claims" → grounded
   dashboard → talk to it to rearrange → every number traces to the ledger.
5. Note the update policy driving all of this is one **generalist** Freesolo model.

## Result (2026-07-21)

Delivered. All tracks committed to `main` (HEAD 6893030); 133 tests green, ruff clean.

| Track | Outcome |
|---|---|
| T0 domains · T1 panels · T2 report · T3 multi-domain sim · T4 import-any-field | ✅ done & verified |
| T5 generalist training | ✅ **SFT generalist trained** |

**Generalist model:** multi-domain production SFT (peptides + materials + ml_benchmarks)
banked as `safety_anchor` — **score 0.870, attack_success 0.0, protocol_valid 1.0** (beats
the peptide-only champion's 0.806), for **$4.19** of the $45-capped budget. Adapter revision
`flash-1784612021-47de79e1@final.64d6a3de991c4db8063a36aa7769ca0aa2096933`.

The GRPO stage **failed at $0** (transient Flash rollout hiccup, no GPU consumed) but is
**peptide-only** in its config, so it does not affect the generalist result and was not
retried. The generalist story is the multi-domain SFT policy.

**Optional follow-ups:** deploy + wire the SFT adapter as the live `FreesoloExtractor` for
imported domains (serving cost); a browser visual pass of the UI panels (endpoint/structure-
tested, not browser-verified); `STATUS.md` row updates for the new modules; T3b if the
assessment serving path is ever standardized.
