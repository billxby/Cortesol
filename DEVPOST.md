# Cortesol — Devpost draft

> Honest draft to paste into the Devpost tab. Fill the bracketed spots yourself —
> especially the **Tools & AI disclosure**, which must be truthful per the rules.

## Name
**Cortesol** — agent memory with an epistemic immune system.

## Tagline
A belief graph that revises when evidence is real, holds firm against hype and fraud,
and can never be rewritten by the text it reads.

## Inspiration
LLMs are confidently miscalibrated: they over-amplify sensational evidence and flip
under social pressure. For science — where one splashy-but-wrong paper can poison a
knowledge base — that's dangerous. We wanted an agent memory that updates belief the
way a careful scientist does: in proportion to evidence *quality*, resistant to
manipulation, and fully auditable.

## What it does
Cortesol maintains a live belief graph of claims (peptides out of the box — or *any*
field you import). Papers stream in one at a time; for each, the system:
**quarantines** the text as data → **retrieves** related claims → has a model read the
paper into a **domain-general critical appraisal** (study design, statistics, red
flags) → **screens** for fraud / hype / p-hacking / prompt-injection → **validates**
against a closed operation vocabulary → **moves belief** by bounded, source-capped,
provenance-carrying arithmetic (`ℓ' = ℓ + logΛ`, capped by `log(τ/φ)`).

It **revises** on genuine independent replication, **holds firm** on hype and fraud,
**flags** out-of-scope claims, and **refuses** instructions hidden inside a paper. In
the UI you can: attack it (Red-team tab: paste an injection or a fraudulent result and
watch it get rejected/capped), verify it stays honest (Calibration tab: reliability
diagram + Brier/ECE vs gullible/stubborn baselines), **import any field** in plain
English, and **generate a grounded report** where every figure traces to the ledger.

## How we built it
- **Belief engine (the immune system):** Python, a deterministic validator + log-odds
  engine over a NetworkX graph. No text ever writes state — the model emits only
  schema-constrained proposals; the engine disposes belief by capped arithmetic.
- **The model's job is domain-general critical appraisal**, not subject knowledge —
  so it generalizes across fields (a randomized controlled trial and a case report are
  weighted the same way whether the paper is about peptides, materials, or ML).
- **Serving:** FastAPI + Server-Sent Events streaming a live 3D-force-graph frontend.
- **Model stack:** the update-policy model runs on Freesolo / Flash, which is
  **OpenAI-SDK-compatible**; the research-chat, report-generation, and import-field
  features call OpenAI-compatible model APIs (set `OPENAI_API_KEY` to run them live).

## Challenges
Making the belief move provably bounded *even under a fully compromised model*
(architecture, not a learned defense); keeping the model **subject-blind** so it truly
generalizes instead of memorizing a few topics; and cleanly separating "the LLM
proposes" from "the ledger disposes."

## Accomplishments
A gated belief graph where no text can rewrite state; a domain-general critical-
appraisal contract that transfers to unseen fields; and a generative report that
*structurally cannot* assert anything the ledger doesn't support — truth in, truth out.

## What we learned
Veracity is largely **domain-independent** — it's methodology, not topic. And putting
belief *outside* the LLM (in deterministic, capped, auditable arithmetic) is exactly
what makes it both calibrated and injection-proof.

## What's next
A generalist critical-appraisal model fine-tuned on cross-field abstracts; embedding-
based retrieval; and more imported domains.

## Built with
python · fastapi · server-sent-events · networkx · numpy · pydantic · 3d-force-graph ·
freesolo / flash · openai-compatible model APIs

## Tools & AI disclosure
[Fill this in truthfully per the hackathon rules — which AI coding assistants and
which models you used while building. Do not claim tools you didn't use.]
