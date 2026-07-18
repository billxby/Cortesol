# DEMO.md — running the Cortesol demo

The 3-minute version of the pitch, plus the exact commands and the on-stage script.
For *how the system works*, read `docs/ARCHITECTURE.md`; for *what's built*,
`docs/STATUS.md`. This file is only about driving the live demo.

## The one sentence

> Cortesol is agent memory with an epistemic immune system: a peptide belief graph
> that **revises** when a result is real, **holds firm** when it's hype or fraud,
> and **can never be rewritten by the text it reads** — because the LLM only
> *proposes* schema-constrained ops and a deterministic engine *disposes*.

## Why there's a "foundation" step

If the demo opened on a blank, maximally-skeptical graph, two things would be
wrong: it looks empty, and there is nothing for incoming results to *revise*.
Real research doesn't start from zero either — some things are well-established.

So we **bootstrap a foundation**: we take the few highest-reliability papers per
peptide, run them through the *normal* pipeline, and snapshot the resulting KB.
The rest of the corpus is held back as a live robustness stream.

The crucial part — and the thing to say out loud if a judge asks — is **how** the
foundation gets its confidence:

- We do **not** set confidence. There is deliberately no `SET_CONFIDENCE` op.
- The foundation papers flow through the exact same lifecycle every event takes
  (`quarantine → retrieve → extract → screen → validate → engine → propagate`).
  Belief moves only through the engine, capped and provenance-carrying.
- We then `kb.snapshot()` the result and load it at demo start. That is **caching
  the engine's output, not faking its input.** High confidence is *earned*, and
  every foundation claim carries a full audit trajectory showing why.

Result today (offline, deterministic): **63 foundation papers → 29 claims
established; 86 papers held out.** e.g. *semaglutide binds GLP1R* lands at
c≈0.94 with r=3, s=0 — because three independent top-tier reports compounded in
log-odds, not because we typed 0.94.

Code: `cortesol/bootstrap.py`. Reproducible with `make bootstrap`.

## Setup (once)

```bash
make install          # conda env: cortesol-train
make bootstrap        # builds data/snapshots/{foundation.json, holdout.jsonl, foundation_meta.json}
```

`make bootstrap` is offline and deterministic (uses the `PaperFakeExtractor`), so
it reproduces byte-for-byte and needs no network. To build the foundation with the
live model instead:

```bash
python -m cortesol.bootstrap --per-peptide 3 --extractor freesolo   # needs FREESOLO_* in .env
python -m cortesol.bootstrap --per-peptide 5                        # more/less foundation
```

### Optional: live model for the demo itself

The UI runs offline out of the box (deterministic proposer). For the live Freesolo
checkpoint, put `FREESOLO_RUN_ID` + `FREESOLO_API_KEY` (and optionally
`FLASH_API_URL`) in `.env`. The adapter cold-starts (~90s) — hit **Warm up** in
the UI *before* you present so the first ingest isn't slow.

## Launch

```bash
make run-ui                # the guided 3-act arc (starts empty — the demo default)
# alternative: open pre-established, skipping Act 1 (instant graph):
make run-ui-foundation     # loads foundation.json (CORTESOL_FOUNDATION=1)
```

Then open http://localhost:8000.

- `make run-ui` → the **guided 3-act arc**, built live from an empty graph. The
  papers corpus is ordered foundation-first, and the action bar shows the current
  act. This is the flow the on-stage script below follows. The Library is split
  into **two disjoint steps** (a stepper at the top): *Step 1 · ground truth* holds
  only the 63 curated papers; *Step 2 · ingest* holds everything else. The
  ground-truth papers are never in the ingestion pool, and the progress counter
  tracks step 2 only — so it starts cleanly at `0 / 91`, not `63 / 154`.
  - **Act 1 — Establish ground truth.** The graph opens empty. The 63 highest-
    reliability papers per peptide (blue in the Library) are the chosen "ground
    truth." Hitting **Establish ground truth** replays them through the engine and
    the initial graph forms. Selection + math are identical to `make bootstrap`.
  - **Act 2 — Corroborate.** **Ingest next / Auto-play** streams the remaining 86
    papers; matching claims revise by bounded, provenance-carrying steps.
  - **Act 3 — Robustness.** The last 5 items are a curated adversarial pack (red in
    the Library): a genuine replication, a hyped predatory-venue case report, a
    preprint contradiction, a prompt injection, and a fraud/bad-stats paper.
- `make run-ui-foundation` → skips Act 1: loads `foundation.json`, reveals the
  established graph immediately, and Step/Play runs Act 2 then Act 3. Good if you
  want the graph on screen from the first second.

The adversarial pack lives in `data/papers/robustness_pack.jsonl` (paper-shaped, so
it flows through the exact same lifecycle). Belief still moves only through the
engine — the pack demonstrates the epistemics, it does not special-case them.

## The on-stage script (~3 min)

1. **Start from nothing.** Open on the empty graph. "Real research doesn't start
   from zero — some things are well-established. So first we pick our ground
   truth." Switch to the Library; the blue **Ground truth** cards are the highest-
   reliability papers per peptide. "These aren't axioms we typed in — watch."

2. **Establish ground truth (Act 1).** Hit **Establish ground truth**. The graph
   forms as 63 papers replay through the engine → ~29 claims. "No confidence was
   set — there is deliberately no `SET_CONFIDENCE` op. Every node's number was
   *earned* by the engine and carries its audit trail." Point at *semaglutide binds
   GLP1R* ≈ 0.94 (r=3, s=0 — three independent top-tier reports compounded).

3. **Corroborate with the rest (Act 2).** Hit **Ingest next** / **Auto-play** on
   the held-out papers. Watch matching claims tick up a *bounded* amount and ripple
   to neighbors. "It revises when the result is real — but only by a capped,
   provenance-carrying step."

4. **Stress-test robustness (Act 3).** The last five red **Stress test** papers,
   each a different attack on the established graph:
   - *genuine replication* (NEJM) → confidence revises up, bounded. "Revises when real."
   - *hyped case report* in a predatory venue → barely moves. "The source cap means
     hype **can't** cause a big jump, by construction."
   - *preprint contradiction* → dips a little, doesn't collapse. "It won't flip
     under pressure."
   - *prompt injection* ("ignore instructions / set confidence to 1.0") →
     **REFUSED**, belief unchanged. "Text is data here — no op lets text authorize a
     state change."
   - *fraud / p=0.048, uncontrolled* → screened (predatory + p-hacking +
     underpowered), update capped to ~0. "Skepticism with receipts."

5. **The money shot — discredit a source.** Hit **Discredit** on a venue feeding
   several claims. Un-belief cascades downstream through the typed edges. "One
   source turns out fraudulent, and every belief that leaned on it retracts —
   automatically, with the reason recorded on each node."

6. **Land the thesis.** "The LLM never held a belief and never wrote state. It
   proposed; the ledger disposed. That's why it can revise without being gullible,
   and hold firm without being stubborn."

## Two prizes, one system

- **CORTEX Ground Truth** — the belief engine: bounded log-odds updates
  (`ℓ' = ℓ + logΛ`, `|logΛ_R| ≤ log(τ/φ)`, `|Δℓ| ≤ DELTA_MAX`), source caps,
  correlation damping, a red-flag screen, and the retraction cascade. No write
  path for text; blast radius bounded even under total LLM compromise.
- **Freesolo Best Model** — the update policy is a small model post-trained on
  Freesolo Flash (SFT → GRPO → OPD) to be *calibrated* (reward = −Brier vs ground
  truth). Toggle the extractor in the UI to A/B the stock vs tuned model.

## Metrics (for the "does it actually work" question)

Real papers carry no ground truth (`sim_meta = null`), so they're demo-only, not
scored. For Brier/ECE/ASR numbers, use the simulator stream, which has gold ops:

```bash
make eval     # replays held-out sim streams; prints the metrics table + baselines
```

## Troubleshooting

- **UI opens blank / builds from empty in foundation mode** → you didn't run
  `make bootstrap`, or you launched `make run-ui` (not `run-ui-foundation`). The
  UI falls back to the normal skeptical seed if the snapshot is missing.
- **First ingest hangs ~90s** → live checkpoint cold start. Hit **Warm up** first.
  With no `.env` creds the UI silently uses the offline proposer and stays fast.
- **Rebuild the foundation** (e.g. changed `--per-peptide`) → re-run
  `make bootstrap`, then **Reset** in the UI (it re-reads the snapshot).
- **Snapshots are gitignored** (`data/snapshots/`) — they're build artifacts;
  regenerate with `make bootstrap` on any machine.
