# Components — how the work splits up

Guidance, not law. Cortesol divides cleanly into **three logical areas**. They map
naturally onto three branches if you want to parallelize, but split however suits
you — the point is that the seam between them is small and stable, so people can
work independently without stepping on each other.

The seam is the **shared contracts** in `cortesol/core/` (already written):
`schema`, `ops`, `domain`, `config`, `context`, `results`, `kb`, `mathx`. As long
as everyone imports these and nobody changes them casually, the three areas plug
together. Treat those files as the interface; if one genuinely needs to change,
say so out loud so the other areas don't break under you.

---

## Area A — Ground-Truth Engine (the ledger)

The deterministic epistemic core. This is the substance of the CORTEX story and it
holds the guarantee that text can't rewrite belief. **No LLM, no network, fully
deterministic and unit-testable.**

Owns:
- `core/engine.py` — log-odds updates, source caps, n_eff damping
- `core/validator.py` — the gate (enumerated ops, bounded step, provenance)
- `core/propagate.py` — TruthFinder + typed belief propagation + retraction cascade
- `ingest/quarantine.py` — untrusted-text handling / spotlighting
- `ingest/screen.py` — the deterministic red-flag battery
- (stewards the contracts in `core/`, since the data model and op vocabulary live
  closest to the engine)

Consumes: nothing outside `core/`. This area can be built and tested entirely on
its own with hand-written ops and fixtures.

Reference notes: `../Research/Confidence Math.md`, `Graph Propagation and GNNs.md`,
`Belief Revision and Truth Maintenance.md`, `Fraud and Hype Signals.md`.

## Area B — Model & Simulator (the proposer + the factory)

The LLM update policy and everything that trains it. This is the Freesolo story.

Owns:
- `ingest/extract.py` — the model that reads a Context and proposes ops
- `sim/world.py`, `sim/events.py`, `sim/gold.py` — the simulator: a ground-truth
  peptide world → event stream (7 classes) → deterministic gold ops
- `train/make_sft.py`, `train/environment.py`, `train/configs/*.toml` — SFT → GRPO
  → OPD on Freesolo Flash

Consumes: the contracts in `core/` (its ops must be valid ops; its gold is
expressed in the op vocabulary). For GRPO, `train/environment.py` imports the
**real** engine + validator from Area A so the reward is scored against true belief
state — so it depends on Area A being reasonably stable, which happens late.

Reference notes: `../Fine-Tuning Plan.md`, `Prompt Injection Defense.md`,
`Fraud and Hype Signals.md`.

## Area C — Cohesion (the arena)

The orchestrator that wires A and B together, the harness that proves it works, and
the demo. This is where the two stories become one runnable system.

Owns:
- `pipeline.py` — the per-event lifecycle orchestrator (the glue)
- `retrieval.py` — build the bounded Context for the extractor
- `adapters/cortex.py` — the only file that knows CORTEX's data format
- `eval/*` — replay, baselines (gullible/stubborn/frontier), metrics (Brier/ECE/
  ASR), red-team battery
- `ui/*` — FastAPI + SSE + vis-network live belief graph

Consumes: both other areas, but always **through interfaces**, so it never blocks
on them. The pipeline runs today against `FakeExtractor` (Area B) and stubbed
engine calls (Area A), and swaps in the real ones as they land.

Reference notes: `../System Architecture.md`, `Graph Propagation and GNNs.md` §5,
`../Prize Strategy.md` (the demo script).

---

## How the areas plug together

```
   Area B                Area C                 Area A
  extract  ──ProposedOps─► pipeline ──► validate + engine + propagate
     ▲                        │                    │
  Context ◄── retrieval ──────┘             belief graph (KB)
                              ▲                    │
                              └──── snapshot / audit / SSE ──► UI
```

Data flows in the frozen types: `RawEvent` → `Evidence` → `Context` →
`ProposedOps` → `ValidationResult` → `Delta`/`EventResult`. Nobody needs to know
how the *other* area implements a step — only the type that crosses the boundary.

## Working notes (the shared rules)

These are the few things all three areas rely on. They're spelled out in
`../CLAUDE.md`; the load-bearing ones:

1. Belief moves only through Area A's engine. Never set `claim.ell` elsewhere.
2. Untrusted text is data — quarantine it; a model rationale never authorizes
   anything. Only `ProposedOps.ops` reach the validator, never `.think`.
3. Never leak ground truth: the extractor and engine only ever see
   `event.untrusted_view()`; `sim_meta` is for the eval harness alone.
4. Invalidate, never delete.
5. Import constants from `core/config.py` and peptide specifics from
   `core/domain.py` — no magic numbers.
6. Keep the shared contracts in `core/` stable; announce changes.

## Unblocking each other (fakes)

Each area ships a trivial fake of its interface so the others don't wait:
- Area B → `FakeExtractor` (canned ops) so A and C can run the loop with no model.
- Area A → hand-written ops + a reference apply, so B's env and C's pipeline have
  something real to call.
- Area C → a stub retrieval that returns "all claims" for tiny sims, so B's env can
  build prompts before real retrieval exists.

That's what makes three-way parallel work actually parallel: build against the
interface, not the implementation.
