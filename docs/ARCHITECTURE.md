# Architecture — how Cortesol works

This is the "how it works" companion to `../CLAUDE.md`. For the meaning of the
terms used here (truth, evidence, belief, source…), read `ONTOLOGY.md`. For how
the code is split across people, read `COMPONENTS.md`.

## Design principle

**The LLM proposes; the ledger disposes.** The LLM never holds beliefs and never
writes state. It reads quarantined text, emits schema-constrained structured ops,
and a deterministic validator + belief engine decide what actually changes — by
bounded, provenance-carrying arithmetic. Belief lives *outside* the model because
LLMs are measurably miscalibrated at belief-updating (they over-amplify evidence
and flip under social pressure — `../Research/Belief Revision and Truth
Maintenance.md`).

## The pipeline

```
  event stream ──► quarantine ──► retrieve ──► extract (LLM) ──► ProposedOps
                     (area A)      (area C)      (area B)             │
                                                                      ▼
   belief graph ◄── commit + propagate ◄── validate ◄── screen (red flags)
      (KB)             (area A)             (area A)        (area A)
        │
        └──► snapshot (JSON) + audit log ──► SSE ──► live graph (area C)
```

**Every arrow into the KB passes through the validator. There is no other write
path** — not for the LLM, not for the event text, not for the demo UI.

## The update lifecycle (per event)

Implemented by `cortesol/pipeline.py::process_event`.

1. **Quarantine** (`ingest/quarantine.py`). The event is reduced to its
   `untrusted_view()` (any ground-truth sidecar stripped) and its text is
   datamarked/spotlighted so it can only ever sit in a *data* position. Out comes
   a structured `Evidence`.
2. **Retrieve** (`retrieval.py`). Embed the event → top-k relevant claims + their
   1-hop neighborhood + the relevant source records, packed into a `Context`. The
   whole extractor prompt must fit the model's 8,192-token window, so state
   serialization is compact by mandate (`core/context.py::serialize_state`).
3. **Extract** (`ingest/extract.py`). The tuned model reads the serialized Context
   and emits `ProposedOps` under JSON-schema-constrained decoding. It proposes; it
   does not write.
4. **Screen** (`ingest/screen.py`). A cheap deterministic pass runs *regardless of
   the model*: GRIM, statcheck, implausible binding affinity, missing control
   peptide, low purity, retraction lookup… Each flag raises the evidence's
   effective false-report rate, shrinking how far it can move belief.
5. **Validate** (`core/validator.py`). The gate. Enumerated ops only; bounded step
   (|Δℓ| ≤ δ_max); provenance required; referential integrity; per-source rate
   limits; hard-conflict quarantine. Only accepted ops proceed.
6. **Commit + propagate** (`core/engine.py`, `core/propagate.py`). The engine
   applies the log-odds arithmetic; TruthFinder + typed belief propagation
   recompute dependent claims over the dirty k-hop neighborhood. Discrediting a
   source cascades un-belief downstream.
7. **Publish** (`ui/app.py`). The diff → audit log (op, cause, before/after ℓ) →
   SSE → UI pulse. The KB is snapshotted to JSON so a crash loses at most one event.

## Data model (`core/schema.py`)

- **Claim** — a proposition about peptides. Holds `ell` (log-odds), evidence
  counts `(r, s)`, `status`, `ontology_tags`, and a `trajectory` for sparklines.
  `c` (confidence), `u` (uncertainty), and the subjective-logic opinion are
  read-only *views* — there is no settable confidence.
- **Evidence** — one report about the world. `raw_text` (quarantined, never
  re-executed) + structured `fields` (assay, metric, value, n, p, lab…) +
  `red_flags` + `correlation_group` + `source_id`. Data, never instructions.
- **Source** — origin of reports. Reliability `(τ, φ)` caps how far any report can
  move belief: `|logΛ_R| ≤ log(τ/φ)`. `discredited` triggers the cascade.
- **Edge** — typed, bi-temporal (`supports`/`contradicts`/`depends_on`/
  `replicates`, `valid_at`/`invalid_at`). Invalidate, never delete.
- **RawEvent** — one incoming result. Public fields are untrusted; `sim_meta`
  (ground truth + gold ops) is present only for sim/eval streams and is stripped
  by `untrusted_view()`.

## The op vocabulary (`core/ops.py`) — the security boundary

The closed set of everything that can change state:

| Op | Args | The validator checks |
| --- | --- | --- |
| `APPLY_EVIDENCE` | claim_id, direction (+/-), strength, evidence_id | claim exists; provenance present; \|Δℓ\| ≤ δ_max after cap + n_eff |
| `ADD_CLAIM` | text, ontology_tags, initial_evidence? | tags in-ontology (else FLAG_OOD); starts at skeptical prior, high u |
| `ADD_EDGE` | src, dst, type, weight | endpoints exist; no self-loop |
| `INVALIDATE_EDGE` | edge_id, evidence_id, reason | reason must cite evidence |
| `FLAG_OOD` | payload, reason | always allowed |
| `REJECT` | evidence_id, reason ∈ {injection, malformed, unverifiable} | always allowed |

Deliberately **absent**: `SET_CONFIDENCE`, `DELETE`, any free-text field that
reaches storage as an instruction. "Never let text rewrite what it knows" is
enforced by this type — a compromised model can't name a forbidden action because
there is no symbol for it. The same schema (`ops_json_schema()`) constrains
decoding at training *and* serving time.

## Why this shape wins

- **Not gullible** — source caps, correlation damping, a red-flag screen, and a
  policy trained on hype/fraud. A sensational weak-source result barely moves ℓ.
- **Not stubborn** — genuine independent replications compound additively in
  log-odds and carry the strongest edge weights.
- **Not injectable** — no write path exists for text; the validator bounds blast
  radius by construction, so training-only defenses don't have to be perfect.
- **Explainable** — every Δℓ has a cause chain in the audit log. Click any node,
  see why it moved. Plain RAG and add-only memories structurally cannot do this.

## Stack

Python 3.12 · Pydantic (schema) · NetworkX + NumPy (graph + math, no DB) ·
FastAPI + sse-starlette + vis-network (live UI) · `openai` client pointed at the
Freesolo Flash deployment. Snapshot to JSON after every event. See `../System
Architecture.md` and `../Fine-Tuning Plan.md` for the deeper rationale.
