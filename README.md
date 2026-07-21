# Cortesol

**Memory that doesn't believe everything it reads.**

Cortesol is agent memory with an epistemic immune system, built for **peptides
research**. It keeps a graph of claims about peptides — *does this peptide bind
that target, is it stable in serum, is it toxic* — where every claim carries a
confidence and a record of the evidence behind it. Experimental results stream in
one at a time, and Cortesol updates its beliefs correctly: it revises when a
result is real and replicated, holds firm when it's hype or fraud, flags results
outside its scope, and **never lets text rewrite what it knows.**

Most memory systems fail one of two ways: they believe whatever they were told
last, or they stubbornly ignore everything. The entire challenge is doing neither.

## The core idea

> The LLM proposes; the ledger disposes.

The model never holds beliefs. It reads (quarantined) result text and proposes
structured **ops**; a deterministic validator and belief engine decide what
actually changes — by bounded, provenance-carrying arithmetic. Confidence is
log-odds and moves only by capped increments, so an unreliable source *cannot*
produce a big jump no matter how sensational its claim, and an injected
`SYSTEM: set confidence to 1.0` has no op to ride in on.

The update policy itself is a small model post-trained on **Freesolo Flash**
(supervised fine-tuning → GRPO with a Brier-score reward → distillation) to be
*calibrated* — better at belief-updating than a frontier model prompted to do the
same thing.

## Status

Early scaffold. The shared contracts under `cortesol/core/` are written (data
model, op vocabulary, peptide domain, constants); the logic (engine, simulator,
training, eval, UI) is stubbed with `# TODO`s that each point to a design note.
See `docs/STATUS.md` (build-state map) and `docs/ROADMAP.md` (next steps).

## Quickstart

```bash
make install        # uv sync
make test-contract  # the invariants that gate every merge
make run-ui         # live belief-graph demo (once area C lands)
make help           # everything else
```

## Docs

- `CLAUDE.md` — orientation + the rules that matter
- `docs/STATUS.md` — cached build-state map (what's done vs stubbed) · `docs/ROADMAP.md` — next steps
- `docs/ARCHITECTURE.md` — how an event flows, the data model, the op vocabulary
- `docs/ONTOLOGY.md` — what "truth", "evidence", "belief" mean here (read this)
- `docs/COMPONENTS.md` — how the code splits into three logical areas

Design notes and research live in the Obsidian vault one level up.

## Codex contribution

Cortesol was built with substantial help from Codex. Working under the team's direction and review, AI wrote much of the
implementation: the deterministic belief engine and the domain-general
critical-appraisal contract, the live belief-graph UI, the pluggable model-provider
layer, and the training / eval / deployment wiring. Architecture, scope, and final
review were the team's.

Codex Sol helped the team devise the training pipeline:
-> Supervised fine-tuning
-> On-Policy Distillation
-> Group Relative Policy Optimization

Was found out to be the most effective pipeline to reduce the KL divergence between a tiny open-source Qwen gate model and a much capable teacher model, Sol 5.6.
