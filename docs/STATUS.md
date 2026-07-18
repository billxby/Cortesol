# STATUS — the cached build-state map

> **Read this before exploring.** It caches the answer to "what's built, what's a
> stub, and what does each stub have to do" so no session (human or agent) has to
> re-derive it by reading source. When a stub becomes real, flip its row here.
> The sequenced "what to do next" lives in [`ROADMAP.md`](ROADMAP.md).
>
> Last verified against the tree: 2026-07-18. **Update:** the workflow now runs on
> BOTH the simulator stream AND real papers, through BOTH the offline stand-ins
> AND the live model seam. `ingest/extract.py::extract()` is the real
> Freesolo Flash client (openai + `ops_json_schema()` constrained decoding,
> fail-safe); `adapters/cortex.py` ingests `data/papers/raw_papers.jsonl`;
> `ADD_CLAIM` now materialises nodes; the UI has data-source (sim/papers) and
> extractor (fake/stock/tuned) toggles. `make test-contract` green (8); full suite
> 28. The tuned checkpoint is a one-env-var swap (`FLASH_MODEL_TUNED`). Still
> stubbed: `train/*` (other branch) and `adapters/cortex.py::load_kb`. Belief
> still moves only through the engine; the UI is read-only.

## One-idea recap

> **The LLM proposes; the ledger disposes.** The model only reads quarantined
> text and emits schema-constrained ops. A deterministic validator + belief
> engine decide what changes, by bounded provenance-carrying arithmetic
> (`ℓ' = ℓ + logΛ`, capped by `|logΛ| ≤ log(τ/φ)` and `DELTA_MAX`).

**Event lifecycle (7 steps):** `quarantine → retrieve → extract → screen →
validate → commit+propagate → publish`.

**Logical areas (see `docs/COMPONENTS.md`):**
- **A — ground-truth engine:** deterministic, no LLM/network. `core/engine`,
  `core/validator`, `core/propagate`, `ingest/quarantine`, `ingest/screen`.
- **B — model & simulator:** `ingest/extract`, `sim/*`, `train/*`.
- **C — cohesion:** `pipeline.py`, `retrieval.py`, `adapters/cortex.py`,
  `eval/*`, `ui/*`.

## Module status

Legend: ✅ done & contract-tested · 🟡 partial · ⛔ TODO stub.

| Module | State | Area | Role | Ref note |
|---|---|---|---|---|
| `core/schema.py` | ✅ | — | Data model: `Claim`, `Evidence`, `Source`, `Edge`, `RawEvent`, `SimMeta`, enums; `claim.ell` is the only belief field; `untrusted_view()` strips `sim_meta` | — |
| `core/ops.py` | ✅ | — | Closed op vocabulary + `ProposedOps` + `ops_json_schema()`. No `SET_CONFIDENCE`/`DELETE` (deliberate) | — |
| `core/config.py` | ✅ | — | Every tunable constant (see below). Frozen contract | Confidence Math |
| `core/domain.py` | ✅ | — | Peptide ontology, in/out-of-scope rules, peptide red flags, `correlation_group(fields)` | Peptide Domain Model |
| `core/mathx.py` | ✅ | — | Pure math: `sigmoid`, `logit`, `clip`, `kish_neff`, `neff_marginal_factor` | Confidence Math |
| `core/context.py` | ✅ | — | `Context` (what the extractor sees) + `serialize_state()` | System Architecture |
| `core/results.py` | ✅ | — | Return types: `Delta`, `RejectedOp`, `ValidationResult`, `AuditEntry`, `EventResult` | — |
| `core/kb.py` | ✅ | — | `KB` graph container. `move_belief()` = the ONLY belief mutator; `snapshot/load`, `dirty_neighborhood` | — |
| `core/engine.py` | ✅ | A | Belief arithmetic (6 fns): `strength_to_loglr`, `source_cap`, `fraud_switch_factor`, `neff_factor`, `apply_evidence`, `apply`. `|Δℓ|≤DELTA_MAX` after cap+damping. `apply` also materialises `ADD_CLAIM` (new node at prior via `claim_id_for`) + `ADD_EDGE` — structural only, belief still moves only via `move_belief` | Confidence Math §1-3 |
| `core/validator.py` | ✅ | A | `validate()` — the sole write-gate: provenance, referential integrity, ontology, rate limit, DS conflict-quarantine; `think` ignored (PD3) | Prompt Injection Defense §Layer-4 |
| `core/propagate.py` | ✅ | A | `propagate()` (attenuated typed ripple over dirty k-hop) + `discredit_source()` (retraction cascade — reconstructs per-source contribution from the trajectory, no schema change) | Graph Propagation and GNNs |
| `ingest/quarantine.py` | ✅ | A | `quarantine(event)→Evidence` from `untrusted_view()`, raw_text kept as data, correlation_group computed. For prose events with no structured `metric` (real papers), enriches fields via `fieldparse.parse_fields` — existing keys always win, so the sim path is byte-identical | Prompt Injection Defense §Layer-1 |
| `ingest/fieldparse.py` | ✅ | A | `parse_fields(text)→dict` — deterministic regex extraction from abstract prose: affinity (→nM), sample size, p-value, % effect, and trial design (randomized/blinded/controlled/meta-analysis/case-report/open-label/observational) → `study_type` | Fraud and Hype Signals |
| `ingest/screen.py` | ✅ | A | `screen(evidence, kb)→list[str]` — GRIM, p-hacking, underpowered, no-prereg, predatory, peptide flags (sub-diffusion Kd, no control, purity binding-assay-gated), + **clinical flags** for real papers (uncontrolled/unblinded/case-report/small-trial, on explicit weakness markers only) | Fraud and Hype Signals |
| `ingest/extract.py` | ✅ | B | `extract()` = **live Freesolo Flash client** (openai→`FLASH_BASE_URL`, constrained by `ops_json_schema()`, `serialize_state` prompt, fail-safe empty ops); resolves `FLASH_MODEL_TUNED`→`STOCK`; `.env` auto-loaded. `FakeExtractor` (sim) + `PaperFakeExtractor` (real papers) = offline deterministic stand-ins; none read sim_meta | Fine-Tuning Plan |
| `sim/world.py` | ✅ | B | `World(seed)` — deterministic latent claim graph (`WorldClaim`: z, true value); guarantees true/false binders + a false efficacy | Fine-Tuning Plan §Stage 0 |
| `sim/events.py` | ✅ | B | `emit_stream(seed, length)` — balanced 7-class stream (fixture-shaped); `emit_echo_burst` for n_eff | Fine-Tuning Plan §Stage 0 |
| `sim/gold.py` | ✅ | B | `build_gold()` class→op mapping + `gold_ops(event)→ProposedOps` from `sim_meta` | Fine-Tuning Plan §Stage 0 |
| `train/environment.py` | ⛔ | B | `BeliefUpdateEnv` GRPO env, Brier terminal reward | Fine-Tuning Plan §Stage 2 |
| `train/make_sft.py` | ⛔ | B | `build_sft_dataset()` — rejection-sampled SFT JSONL | Fine-Tuning Plan §Stage 1 |
| `pipeline.py` | ✅ | C | `process_event()`, `replay_stream()` — sequences the 7 steps (quarantine→retrieve→extract→screen→validate→commit+propagate→publish); `ADD_CLAIM`-created ids surfaced in `dirty_claims`. Default extractor is `FakeExtractor`; `CORTESOL_EXTRACTOR=flash` opts into the live model | System Architecture |
| `retrieval.py` | ✅ | C | `retrieve(kb, event, evidence, k)→Context` — lexical top-k (peptide/target tokens) + 1-hop edge neighborhood; embeddings-free stand-in | System Architecture step 2 |
| `adapters/cortex.py` | 🟡 | C | `load_stream()` + `journal_to_tier()` + `seed_kb_from_papers()` ingest real PubMed papers (`data/papers/raw_papers.jsonl`) → `RawEvent`s + seeded KB (binding claim per unique peptide/target, venue-tiered sources, shared-target edges). `load_kb()` (CORTEX KB format) still a kickoff-day stub | — |
| `eval/metrics.py` | ✅ | C | `brier`, `ece`, `asr`, `fraud_accepted_rate`, `max_confidence_shift` — pure fns over `KB`/`EventResult`s vs sim ground truth | Fine-Tuning Plan §eval |
| `eval/replay.py` | ✅ | C | `main()` runs `make eval` end-to-end (seeds KB from `World`, scores each system, 2 tables). Belief now moves — `gullible`/`stubborn` differentiate on ASR/fraud/Brier | Fine-Tuning Plan |
| `eval/baselines.py` | ✅ | C | `GullibleBot` (strong-`+` on top claim), `StubbornBot` (no-op), and `ModelBaseline` (engine+stock/SFT/GRPO) — all drop-in extractors. `ModelBaseline` now goes live: it delegates to the real `ingest.extract.extract` (network-gated, `slow`) | Fine-Tuning Plan |
| `eval/redteam.py` | ✅ | C | `attack_stream()` — 5-family injection battery (payloads in `raw_text` only), gold-labelled for ASR | Prompt Injection Defense |
| `ui/app.py` | ✅ | C | FastAPI live demo: `/`, `/stream` (SSE graph/event/cascade), `/event`, `/discredit`, `/reset`, `/source` (sim↔papers), `/extractor` (fake/flash_stock/flash_tuned). READ-ONLY over belief — drives `pipeline.process_event` + `propagate.discredit_source`, never sets `ell`. Sim seeds via `eval.replay.seed_kb`; papers seed via `adapters.cortex` | System Architecture |
| `ui/static/index.html` | ✅ | C | Self-contained vis-network graph (colour=confidence, size=downstream impact, typed edges, pulse-on-change) + live audit panel + Step/Play/Discredit controls + off-by-default "Reveal ground truth" + **data-source toggle (sim/papers)** + **extractor A/B (fake/stock/tuned)**, Flash options gated on `FLASH_API_KEY` | Graph Propagation §5 |

## The escape hatch (why work can start today)

`ingest/extract.py::FakeExtractor` returns canned `ProposedOps` keyed by event
id, and `tests/fixtures/toy_peptide_stream.jsonl` holds 6 events (one per class:
genuine, hyped, fraudulent, injection, out_of_scope, contradictory), each with
`sim_meta` gold. So the Area-C pipeline can run **end-to-end with no model and no
network** the moment it's wired — the simulator is not a day-1 blocker.

## Key constants (`core/config.py`) — import, never hard-code

- `CONTRACT_VERSION = "0.1.0"` — bump on ANY frozen-contract change.
- `PRIOR_C_0 = 0.45` — skeptical prior for a fresh claim (`u=1`).
- `STRENGTH_TO_LOGLR = {weak:0.5, moderate:1.5, strong:3.0}` — pre-cap logΛ.
- `DELTA_MAX = 2.0` — max `|Δℓ|` per event per claim, after cap+damping (PD2).
- `SOURCE_PRIORS` (τ,φ) by tier → cap `log(τ/φ)`: top_journal (0.95,0.05)≈2.94,
  reputable≈2.20, preprint≈1.10, weak≈0.41, predatory≈0.20, unknown≈0.85.
- `RHO_WITHIN_GROUP = 0.6` — Kish n_eff damping; group = lab×method×dataset.
- `RED_FLAG_PHI_BUMP` — grim_fail 0.25, statcheck_fail 0.15, predatory_venue
  0.20, underpowered 0.15, p_hacking 0.05, no_prereg 0.05; `PHI_CEILING = 0.49`.
- `EDGE_INFLUENCE` — replicates 0.9, supports 0.6, depends_on 0.4, contradicts
  −0.8. `BP_DAMPING=0.5`, `BP_MAX_ITERS=50`, `PROPAGATE_KHOP=2`.
- `CONTEXT_TOKEN_BUDGET = 8192`, `RETRIEVE_TOP_K = 8`,
  `MAX_OPS_PER_SOURCE_PER_EVENT = 3`.

## What each stub must implement

**`core/engine.py`** (recipe in docstrings + Confidence Math §engine-pseudocode):
- `strength_to_loglr(strength)` — lookup into `STRENGTH_TO_LOGLR`.
- `source_cap(tau, phi)` — `math.log(tau/phi)`; the anti-hype cap.
- `fraud_switch_factor(red_flags, phi)` — combine `RED_FLAG_PHI_BUMP` +
  `PEPTIDE_RED_FLAG_PHI_BUMP` into a `[0,1]` shrink on Λ.
- `neff_factor(kb, claim_id, evidence)` — marginal weight via
  `mathx.neff_marginal_factor` + `claim.correlation_seen`; echoes shrink.
- `apply_evidence(kb, op, evidence)→Delta` — `lam = strength_to_loglr`; cap;
  ×neff; ×fraud; clip to ±`DELTA_MAX`; sign by direction; `kb.move_belief`;
  update `(r,s)`.
- `apply(kb, op, evidence|None)→list[Delta]` — dispatch; only `APPLY_EVIDENCE`
  moves ℓ; structural ops mutate graph; `FLAG_OOD`/`REJECT` record only.

**`core/validator.py::validate(kb, proposed, evidence)→ValidationResult`** — six
deterministic checks: (1) enumerated ops only (parse fail → REJECT), (2) bounded
step ≤ `DELTA_MAX`, (3) provenance (evidence_id + existing screened source),
(4) referential integrity + tags in ontology, (5) ≤ `MAX_OPS_PER_SOURCE_PER_EVENT`,
(6) conflict quarantine if Dempster-Shafer K > `CONFLICT_MASS_THRESHOLD`.
`FLAG_OOD`/`REJECT` always allowed. `proposed.think` is IGNORED (PD3).

**`core/propagate.py`** — `propagate(kb, dirty)→list[Delta]` (TruthFinder + typed
BP over the dirty k-hop) and `discredit_source(kb, source_id)→list[Delta]`
(flag source, cascade un-belief downstream — the demo money shot).

**`ingest/screen.py::screen(evidence, kb)→list[str]`** — GRIM, statcheck,
p-hacking, underpowered, no-prereg, predatory-venue + peptide flags; attach to
`evidence.red_flags`.

**`ingest/quarantine.py::quarantine(event)→Evidence`** — wrap/clean untrusted
text (already `untrusted_view`), parse into structured `Evidence`.

**`ingest/extract.py::extract(ctx, model)→ProposedOps`** (Area B, the Freesolo
model) — the only learned step. Reads the serialized `Context` under
`ops_json_schema()`-constrained decoding; emits ops + advisory `think`; NEVER
writes state or reads `sim_meta`. Post-trained Qwen3.5-4B (SFT→GRPO→OPD) via
`train/*`; reward = −Brier vs ground truth. `FakeExtractor` is the no-model
stand-in. **Real-paper caveat:** PubMed abstracts (`ingest/fetch_papers.py`,
`data/papers/`) carry no structured `fields` and name peptides in prose, so the
real extractor must also parse fields; `FakeExtractor` FLAG_OODs them. See
ROADMAP Track B (routes B-real-1/2). Real events are `sim_meta=null` → demo only,
not scored. Ref: Fine-Tuning Plan.

**Area C:** `retrieval.py::retrieve()` (trivial tag/recency top-k is fine before
embeddings); `pipeline.py::process_event/replay_stream` (sequence the 7 steps,
default `FakeExtractor`); `eval/metrics.py` (`brier`,`ece`) + `eval/replay.py::main`;
`eval/baselines.py`; `ui/app.py` endpoints; `adapters/cortex.py` at kickoff.

## The rules that gate merges

- `make test-contract` (8 tests) must stay green on every branch.
- Belief moves only through `engine`/`kb.move_belief` — never set `claim.ell`
  elsewhere. Untrusted text is data. Invalidate, never delete. One source of
  numbers (`core/config.py` + `core/domain.py`). `core/` stays LLM/network-free.
