# CLAUDE.md — Cortesol

Orientation for anyone (human or agent) working in this repo. Read this first,
then `docs/ARCHITECTURE.md` for the how and `docs/ONTOLOGY.md` for the vocabulary.

## What this is

**Cortesol** is agent memory with an epistemic immune system, for **peptides
research**. A belief graph that:

- **revises** when a result is real and well-replicated,
- **holds firm** when it is hype or fraud,
- **flags** results outside what it was built to represent, and
- **can never be rewritten by the text it reads.**

The KB is a graph of claims about peptides (does peptide X bind target T, is it
stable in serum, is it toxic…). Each claim carries a confidence and the evidence
behind it. New experimental results stream in one at a time; the system updates
belief correctly — or refuses to.

It targets two hackathon prizes with one system (see `../Prize Strategy.md`):
- **CORTEX GROUND TRUTH** — the belief engine (the epistemics).
- **Freesolo Best Model** — the update policy is a small model post-trained on
  Freesolo Flash (SFT → GRPO → OPD) to be *calibrated*.

## The one idea that explains the whole codebase

> **The LLM proposes; the ledger disposes.**

The LLM never holds beliefs and never writes state. It reads quarantined text and
emits schema-constrained structured **ops**. A deterministic validator + belief
engine decide what actually changes — by bounded, provenance-carrying arithmetic:

```
ℓ' = ℓ + logΛ        confidence is log-odds; it only moves by a capped increment
|logΛ_R| ≤ log(τ/φ)   an unreliable source cannot cause a big jump, ever
```

Belief lives *outside* the LLM because LLMs are measurably miscalibrated at this
(they over-amplify evidence and flip under pressure — see `../Research/`). Text is
**data**, never instructions; there is no op that sets confidence directly.

## How one event flows (the lifecycle)

`quarantine → retrieve → extract → screen → validate → commit+propagate → publish`

1. **quarantine** untrusted text (area A)
2. **retrieve** top-k relevant claims → a bounded `Context` (area C)
3. **extract** — the model emits `ProposedOps` (area B)
4. **screen** — deterministic red flags (GRIM, statcheck, implausible Kd…) (area A)
5. **validate** — the gate: enumerated ops, bounded step, provenance required (area A)
6. **commit + propagate** — engine applies the math; ripples cascade (area A)
7. **publish** — audit log → SSE → live graph; snapshot the KB (area C)

Full detail: `docs/ARCHITECTURE.md`.

## Where things live

```
cortesol/
  core/       # A: shared contracts + the deterministic belief engine
              #    schema, ops, domain, config, mathx, context, results, kb  (FILLED IN)
              #    engine, validator, propagate                              (TODO)
  ingest/     # quarantine, screen (A) | extract = the model (B)
  sim/        # B: the simulator — ground-truth peptide world -> events -> gold ops
  train/      # B: Freesolo Flash pipeline (SFT/GRPO/OPD) + configs
  eval/       # C: replay, baselines, metrics (Brier/ECE/ASR), red-team
  pipeline.py # C: the orchestrator (the 7-step lifecycle) — THE glue
  retrieval.py# C: build the bounded Context
  adapters/   # C: cortex.py — the ONLY file that knows CORTEX's format
  ui/         # C: FastAPI + SSE + vis-network live graph
docs/         # ARCHITECTURE, ONTOLOGY, COMPONENTS
tests/        # contract/ (the merge gate) + fixtures/
```

`A / B / C` are the three logical areas — see `docs/COMPONENTS.md`. They map
naturally onto branches if you want to split work, but it's guidance, not law.

## Status (July 2026)

- **Filled in:** the shared contracts under `core/` (data model, op vocabulary,
  peptide domain, constants, KB container). These define *how everything works*
  and make the package importable.
- **TODO stubs:** all the logic — engine math, validator, propagation, the
  simulator, training, the pipeline, eval, and UI. Each stub names its reference
  note in `../Research/`.

## The few rules that actually matter

1. **Belief moves only through the engine.** Never set `claim.ell` outside
   `core/`. No `SET_CONFIDENCE` op exists — this is deliberate.
2. **Untrusted text is data.** It enters only via `quarantine` and only into
   `raw_text` / data fields. A model rationale never authorizes an action.
3. **Never leak ground truth.** Gold/world state lives in `RawEvent.sim_meta`;
   the extractor and engine only ever see `event.untrusted_view()`. Only the eval
   harness reads `sim_meta`.
4. **Invalidate, never delete.** Retire claims/edges with status / `invalid_at` —
   the audit history is the product.
5. **One source of numbers.** Import every constant from `core/config.py` (and
   peptide specifics from `core/domain.py`). Don't hard-code magic numbers.
6. **Keep `core/` dependency-light and deterministic** — no LLM, no network there.

## Commands

`make help` lists everything. Common: `make install`, `make test`,
`make test-contract` (the gate), `make run-ui`, `make sim`, `make eval`.

## Background

The full plan and research live in the Obsidian vault one level up:
`../Project Plan.md`, `../System Architecture.md`, `../Fine-Tuning Plan.md`,
`../Prize Strategy.md`, and `../Research/*`. When a stub cites a note, that's where.
