# ROADMAP — what to build next, by owner

> Companion to [`STATUS.md`](STATUS.md) (the build-state map). This is the task
> board: dependency order, who owns what, definition-of-done per module.
> Last updated: 2026-07-17.

## Ordering principle

The 6-event fixture + `FakeExtractor` already unblock the pipeline, so the two
tracks below run **in parallel from day 1** and meet at the frozen `core/`
contracts. Belief only visibly moves once `core/engine.py` exists — so the
engine is the critical path. Nobody edits `core/` contracts without bumping
`CONTRACT_VERSION` and keeping `make test-contract` green.

## Track C — overall coherence (the glue)  ·  owner: Bill

Goal: an event flows all the way through and the KB visibly updates.

1. **`retrieval.py::retrieve()`** — trivial tag/recency top-k → `Context`.
   Dependency-light is fine before embeddings. *Done = returns a `Context` that
   fits `CONTEXT_TOKEN_BUDGET`.*
2. **`pipeline.py::process_event()` + `replay_stream()`** — sequence the 7 steps,
   default `extractor=FakeExtractor`. *Done = `replay_stream` runs the 6 fixture
   events without error and returns `EventResult`s.*
3. **`eval/metrics.py` (`brier`, `ece`) + `eval/replay.py::main()`** — feed
   `sim_meta` gold as ground truth. *Done = `make eval` prints Brier/ECE.*
4. **`eval/baselines.py`** — `GullibleBot` / `StubbornBot` anchors.
5. **`ui/app.py`** — `/`, `/stream` (SSE), `/event`, `/discredit` + mount static.
   *Done = `make run-ui` shows the graph pulsing on the fixture stream.*
6. **`adapters/cortex.py`** — leave stubbed until kickoff (CORTEX format unknown);
   keep it isolated so only this file changes later.

## Track A — ground truth (the deterministic engine)  ·  owner: teammate

Goal: belief moves correctly, or refuses to. All TDD against `tests/unit/`.

1. **`core/engine.py`** — the 6 functions (see STATUS.md). **Critical path.**
   *Done = engine unit test passes + contract tests green.*
2. **`core/validator.py::validate()`** — 6 invariant checks, the sole write-gate.
3. **`core/propagate.py`** — `propagate()` + `discredit_source()` (cascade).
4. **`ingest/screen.py::screen()`** — deterministic red-flag battery, no LLM.
5. **`ingest/quarantine.py::quarantine()`** — `RawEvent → Evidence`.

## Shared — the simulator (`sim/`, Area B)  ·  owner: TBD

Not blocking day 1 (the fixture covers early pipeline + engine work) but needed
before training/eval/demo have real volume. `world.py` (seeded latent truth),
`events.py::emit_stream` (7 balanced classes), `gold.py::gold_ops` (labels into
`sim_meta`). Ref: Fine-Tuning Plan §Stage 0. **Decide owner once the engine +
pipeline skeletons land.**

## Handoff contract between tracks

`pipeline.process_event` (Track C) calls `screen()`, `validate()`,
`engine.apply()`, `propagate()` (Track A), all typed by the frozen `core/`
contracts. Until those land, `FakeExtractor` + a no-op commit path lets the
pipeline run; wire each real call as it lands. This is why the tracks don't block
each other.

## Verification

- `make test-contract` — green throughout (the merge gate).
- `make test` — engine/validator unit tests as Track A fills them.
- `make eval` — Brier/ECE on the fixture once metrics + replay land.
- `make run-ui` — live graph once the UI endpoints land.
