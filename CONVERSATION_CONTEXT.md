---
tags: [handoff, context, hackathon, training, mem0, freesolo]
type: conversation-context
status: active
created: 2026-07-17
thread_id: 019f72d4-201a-7e62-9553-fae764c9a201
workspace: /Users/kuanw/Me/code/Hackathon
---

# Conversation Context — Small-Model Memory Policy Training

## Purpose

This file is a self-contained handoff for continuing the discussion in another task. It preserves the user's requests, the decisions reached, the research conclusions, the artifacts created, the current local state, and the immediate next actions.

It is a working-context export rather than a dump of internal reasoning or tool logs. The exact research citations and implementation detail live in the linked workspace artifacts.

## Copy-paste context for a new discussion

> We are building a hackathon demo inspired by Mem0: a small language model proposes structured memory/belief updates, while a deterministic parser, validator, state engine, and simulator remain authoritative. The shared domain data protocol is being built in a separate workstream; do not invent fields, action names, or serialization.
>
> The research recommendation is a provisional Qwen3.5-4B LoRA student, verified SFT/RFT followed by GRPO, with GLM-5.2 OPD only as a gated ablation. However, the current 24–36-hour hackathon override is intentionally much narrower: one Qwen3.5-4B SFT run on 256 verified examples, 24 optimizer steps, checkpoints at 8/16/24, then deployment and a polished live/cached/replay demo. GRPO, OPD, model bakeoffs, and broad accuracy evaluation are off the critical path.
>
> Mem0 is not a model-distillation training recipe. Its “memory distillation” is inference-time extraction/compression. The 2025 paper used LLM-selected ADD/UPDATE/DELETE/NOOP reconciliation; the current 2026 OSS v3 is ADD-only with deduplication, embeddings, entity links, and hybrid retrieval. Keep paper-era and current behavior separate.
>
> Read `24-Hour One-Shot Training and Demo Runbook.md` first, then `Research/Small Model Training Pipeline Memory.md`, `train/PROTOCOL_HANDOFF.md`, `train/configs/sft-one-shot.toml`, and `train/FLASH_RUN_LOG.md`. The current machine has Python 3.14.6, while Flash 1.0.0 requires Python 3.11–3.12; neither `uv` nor `flash` is installed. No training run has been submitted and no endpoint has been deployed yet.

## User intent

The user wants to train a small model for a hackathon system related to long-term agent memory/belief revision. The model should interpret incoming information and propose state/memory operations, but deterministic code must decide whether those operations are legal and correct.

Priorities changed during the conversation:

1. Initially: understand whether the approach is model distillation.
2. Then: perform a current literature/product review and design the full training pipeline without assuming the shared data protocol.
3. Finally: optimize for a 24-hour demo deadline, with 36 hours available as buffer. Accuracy is secondary; speed, appearance, inference readiness, and a convincing demonstration are primary.

## Conversation chronology

### Request 1 — explain the training

User message:

> Explain to me how training the small model will be? from wht I understand, basically model distillation?

Files supplied:

- `/Users/kuanw/Me/code/Hackathon/Fine-Tuning Plan.md`
- `/Users/kuanw/Me/code/Hackathon/.docs/Post Training Slides.pdf`

Answer reached:

- The plan is not merely distillation.
- Start from a pretrained Qwen small model; do not train from scratch.
- Freeze the base model and train a LoRA adapter.
- Verified SFT is imitation/sequence-level distillation only when a larger model generates candidate targets.
- GRPO is reinforcement learning from environment rewards, not distillation.
- OPD is the explicit token-level teacher–student distillation stage.
- The simulator/validator, not the teacher model, must be the source of truth.
- A conflict was identified between free-form `<think>...</think>` targets and structured decoding. For an action-only protocol, non-thinking structured output is preferred; a rationale must be a protocol-approved separate channel/field.

The original conceptual sequence discussed was:

```text
pretrained Qwen student
→ verified SFT
→ GRPO against deterministic environment
→ optional OPD
→ base model + trained LoRA adapter
```

### Request 2 — swarm literature and Mem0 review

User message:

> Spin up a swarm of agents - look at the recent literature on this topic.
>
> Read about mem0 and make sure to identify these key points:
> - Which model to use as the student
> - Which to use as the teacher
> - Dataset configuration
> - Policy
> - which algos per step
>
> We're still generating the data protocol (how the data will look like), but generate a memory file (md) clearly outlining ALL the necessary papers, products, projects.
>
> We will be using this document to build the entire training pipeline. Do not make any assumptions, another agent is building the shared data protocol.

Three parallel research tracks were started:

1. small-model post-training, distillation, RLVR/GRPO, calibration, and security literature;
2. Mem0 and comparable memory systems/products;
3. student/teacher selection, dataset requirements, policy boundary, algorithms, and Flash platform constraints.

Primary artifact produced:

- `Research/Small Model Training Pipeline Memory.md`

It explicitly labels claims as verified, decisions, gates, protocol-TBD, or inference. It does not define any domain fields, operation names, or serialization.

### Request 3 — hackathon deadline override

User message:

> We're doing a hackathon, this has to be done within 24 hours. Elaborate a plan for me to train, and run inference ready within 24 hours. actually, it's 36 hours but whatever. we need this to be one-shot. Accuracy is not priority. Demo, speed and appearance are more important.

Decision reached:

- The full SFT → GRPO → optional OPD pipeline is no longer the hackathon critical path.
- Perform one paid Qwen3.5-4B LoRA SFT run.
- Do not spend a separate run on 0.8B smoke training.
- Do not run a model bakeoff.
- Do not require teacher-generated rationales.
- Use 256 verified examples, 32 local canaries, and five rehearsed showcase cases.
- Train for 24 optimizer steps with checkpoints at 8, 16, and 24.
- Submit by Hour 5; attempt deployment by Hour 10; lock the demo by Hour 18; freeze/rehearse by Hour 24.
- Treat Hours 24–36 as recovery and rehearsal buffer only.
- Build live, cached-trained, and deterministic-replay modes.
- Cache only genuine responses captured from the actual trained run; never hand-write trained-cache outputs.

Primary artifact produced:

- `24-Hour One-Shot Training and Demo Runbook.md`

### Request 4 — continue operationalizing

User message:

> keep going

Work completed:

- Added a ready-to-edit one-shot Flash TOML.
- Added a protocol handoff checklist that makes no schema assumptions.
- Added a credential-safe Flash run log.
- Added training directory instructions.
- Added explicit ownership between protocol/data, training operations, inference/app, and demo/pitch work.
- Added inference request examples, timeout/retry/validation policy, caching rules, checkpoint selection, stop conditions, and release checklist.
- Updated `.gitignore` for secrets and local Python artifacts.
- Updated the older project/fine-tuning plans with warnings that the one-shot runbook is currently authoritative.

### Request 5 — context export

User message:

> export this entire conversation as a context file for another discussion

This file is that export.

## Core research conclusions

### Is this model distillation?

Only partly:

| Stage | What happens | Distillation? |
| --- | --- | --- |
| Verified SFT/RFT | Student imitates accepted target sequences; targets may be direct deterministic gold or teacher candidates filtered by the validator | Sequence-level distillation only when a teacher generated the accepted target |
| GRPO/RLVR | Student samples actions/episodes and receives deterministic environment rewards | No |
| OPD | Student samples its own completions and is trained toward a stronger teacher's token distribution | Yes, token-level on-policy distillation |
| Mem0 “memory distillation” | Inference-time extraction/compression of useful facts | No weight training |

### Student model

Research pipeline recommendation:

- **Provisional production student:** `Qwen/Qwen3.5-4B`
- **Plumbing-only model:** Qwen3.5-0.8B
- **Cost challenger:** Qwen3.5-2B
- **Capability challenger:** Qwen3.5-9B
- Select the smallest model that passes a protocol-specific equal-budget bakeoff when time permits.

Hackathon override:

- Use Qwen3.5-4B directly.
- Do not run the challengers.
- Qwen3.5 is in Flash's experimental tier, so deploy/reload/inference behavior must be verified early.

Direct precedent:

- AgeMem reports memory-policy GRPO using Qwen3-4B-Instruct, which supports the plausibility of a 4B Qwen-family student but does not prove Qwen3.5-4B will succeed on this protocol/platform.

### Teacher roles

Do not collapse different teacher roles:

| Role | Decision |
| --- | --- |
| Gold authority | Deterministic simulator, parser, executor, and validator only |
| Offline SFT generator | Select by verified yield/downstream student gain; omit entirely when deterministic gold can be serialized |
| Online OPD teacher | Flash-managed GLM-5.2 only as a gated experiment; use only if it beats the student on the target task and OPD improves held-out results |
| GRPO reward | Deterministic environment, never an LLM teacher |
| Evaluation judge | Deterministic metrics first; LLM only for qualitative analysis where mechanics cannot score |

For the one-shot hackathon run, there is no required teacher. Direct deterministic target serialization is preferred.

### Policy boundary

The trainable policy is the LoRA-adapted student's conditional distribution over the next protocol-compliant output given protocol-rendered observable history.

The model is a proposer. It is not:

- the source of truth;
- the database;
- the belief engine;
- the parser;
- the validator;
- the executor;
- the authority on whether its own output is correct.

Required external contracts, owned by the shared protocol/system:

1. deterministic renderer;
2. deterministic parser;
3. deterministic validator/executor;
4. deterministic scorer/oracle;
5. reproducible seeds and provenance.

Write admission and read admission are separate. Similarity retrieval is not authorization to expose a memory.

### Shared protocol boundary

The following remain owned by the separate protocol workstream:

- domain objects and field names;
- operation/action names and arguments;
- input/output serialization;
- identifiers and timestamps;
- provenance representation;
- contradiction, deletion/invalidation, supersession, abstention, rejection, confidence, uncertainty, privacy, and scope representation;
- which errors are recoverable/terminal;
- exact gold representation.

The training workstream may require semantic capabilities but must not invent concrete fields.

Minimal handoff required by Hour 2:

1. `render_input(case)`
2. `serialize_target(gold)`
3. `parse_output(text)`
4. `validate(action, visible_state)`
5. ten canonical conformance examples
6. five frozen judge-facing showcase cases

See `train/PROTOCOL_HANDOFF.md`.

## Mem0 findings

### Mem0 2025 paper

[Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) describes inference-time memory orchestration, not student training.

Paper-era behavior:

- rolling summary plus recent messages;
- LLM extraction of candidate facts;
- retrieval of related memories;
- a second LLM/tool call chooses `ADD`, `UPDATE`, `DELETE`, or `NOOP`;
- GPT-4o-mini used for LLM operations in the reported setup;
- Mem0 graph variant adds Neo4j entities/relations and invalidates contradictory edges rather than hard-deleting history.

Do not assume graph memory is always better; paper results did not show uniform improvement on every single-/multi-hop slice.

### Current Mem0 2026 OSS v3

The [official migration guide](https://docs.mem0.ai/migration/oss-v2-to-v3) and current repository describe materially different behavior:

- single-pass ADD-only extraction;
- no LLM `UPDATE` or `DELETE` reconciliation;
- exact textual/hash deduplication;
- embeddings and entity links;
- semantic retrieval with BM25/entity boosts;
- accumulated old/new memories can coexist;
- external graph-store traversal removed from current OSS interface;
- managed platform temporal/state-resolution behavior must not be claimed as OSS parity.

Mem0 is an architectural/product baseline, not the training recipe.

## Necessary literature and systems

The exhaustive annotated list is in `Research/Small Model Training Pipeline Memory.md`. The most important items are:

### Learned memory policies

- [Memory-R1](https://arxiv.org/abs/2508.19828) — RL-trained memory manager and answer agent; PPO/GRPO; Qwen2.5 3B/7B/14B and Llama 8B; SFT baseline behavior-clones GPT-5 trajectories.
- [AgeMem](https://arxiv.org/abs/2601.01885) — progressive long-term/short-term coordination training and step-wise GRPO; direct Qwen3-4B precedent; some rewards use LLM judges, which should not be copied when deterministic scoring is possible.
- [MemAgent](https://arxiv.org/abs/2507.02259) — long-context, multi-conversation memory-agent RL.
- [A-MEM](https://arxiv.org/abs/2502.12110) — linked/Zettelkasten-style memories and memory evolution.
- [MemRL](https://arxiv.org/abs/2601.03192) — runtime non-parametric memory utility learning.

### Memory systems/products

- [Mem0](https://github.com/mem0ai/mem0)
- [Graphiti / Zep](https://github.com/getzep/graphiti)
- [Hindsight](https://arxiv.org/abs/2512.12818)
- [Letta / MemGPT](https://arxiv.org/abs/2310.08560)
- [LangMem](https://github.com/langchain-ai/langmem)
- [HippoRAG 2](https://arxiv.org/abs/2502.14802)
- MemoryBank, OpenMemory, Cognee, TMEM

### Training and distillation

- [APIGen](https://arxiv.org/abs/2406.18518) — verifier-filtered synthetic action data.
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO critical analysis](https://arxiv.org/abs/2503.20783)
- [On-Policy Distillation / GKD](https://arxiv.org/abs/2306.13649)
- [Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016)
- [Breaking the Tokenizer Barrier](https://arxiv.org/abs/2606.09456)
- [Small Models Struggle to Learn from Strong Reasoners](https://aclanthology.org/2025.findings-acl.1301/)
- MiniLLM, DistiLLM, ReST, LoRA

### Benchmarks

- [LoCoMo](https://arxiv.org/abs/2402.17753)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493)
- [BEAM](https://arxiv.org/abs/2510.27246)
- MemoryAgentBench

### Security/privacy

- [MINJA](https://arxiv.org/abs/2503.03704)
- [Beyond Similarity / MemGate](https://arxiv.org/abs/2606.06054)
- [Hidden in Memory](https://arxiv.org/abs/2605.15338)
- [From Untrusted Input to Trusted Memory](https://arxiv.org/abs/2606.04329)
- [Bad Memory](https://arxiv.org/abs/2607.14611)
- privacy risks in LLM memory systems

## Calibration/reward correction

An important correction was made to the original Fine-Tuning Plan:

- Brier score is strictly proper for probabilities over a fixed, externally specified proposition set.
- Brier alone is unsafe if the model chooses which claims exist or get scored.
- A policy could omit hard claims, shrink the state, or emit a known-false proposition with confidence zero.
- The environment must fix the scored proposition/outcome universe and coverage, normalize variable-size states, or combine explicit correctness/state utility with a bounded proper calibration component.
- RLCR's guarantee involves a correctness term plus a bounded proper confidence term, not arbitrary Brier-only scoring of a model-selected knowledge base.
- Brier/ECE alone do not establish useful discrimination/resolution.

This matters for any later GRPO work. It does not block the current SFT-only hackathon run.

## Full research pipeline versus hackathon override

### Research-quality pipeline

```text
freeze shared protocol + deterministic contracts
→ verified candidate generation/direct gold
→ rejection-sampled LoRA SFT/RFT
→ multi-turn GRPO/RLVR using deterministic environment reward
→ optional OPD branch only when teacher advantage is verified
→ immutable ID/OOD/security evaluation
```

Research comparison branches:

- primary: `SFT → GRPO`
- teacher-guided ablation: `SFT → OPD → GRPO`
- diagnostic only: `SFT → GRPO → OPD`

Final OPD is not automatic polish because it can pull the student away from simulator-aligned behavior.

### Current hackathon pipeline

```text
minimal protocol handoff by H2
→ 256 verified single-run SFT rows by H4
→ publish environment + dry-run + quote
→ one Qwen3.5-4B SFT submission by H5
→ 24 optimizer steps, checkpoints 8/16/24
→ deploy/probe by H10 if possible
→ capture five genuine trained responses
→ live/cached-trained/deterministic-replay demo by H18
→ freeze/rehearse/submit by H24
→ H24–36 recovery buffer only
```

Cut from the hackathon critical path:

- GRPO;
- OPD;
- Qwen model bakeoff;
- 0.8B smoke-training detour;
- long chain-of-thought;
- large synthetic corpus;
- LLM judge;
- broad external benchmarks;
- self-hosted inference/export;
- learned GNN weights, Neo4j, new datastore;
- a second frontend flow;
- a second paid SFT run.

## One-shot configuration

Canonical template: `train/configs/sft-one-shot.toml`

```toml
model = "Qwen/Qwen3.5-4B"
algorithm = "sft"
thinking = false
seed = 42

[environment]
id = "REPLACE_WITH_ORG/REPLACE_WITH_ENV"

[train]
max_examples = 256
max_steps = 24
save_at_steps = [8, 16, 24]
max_context_tokens = 1024
batch_size = 32
learning_rate = 0.0001
lora_rank = 32
lora_alpha = 64
```

This follows Flash 1.0.0 SFT defaults closely. `structured_outputs` is intentionally not in the SFT config because Flash applies that configuration to rollout algorithms. SFT output reliability comes from short/consistent targets, completion-only loss, low-temperature inference, parser/validator enforcement, retry, and fallback.

## Dataset plan for one-shot run

- 256 accepted training rows uploaded.
- 32 local canary rows not uploaded.
- Five exact showcase cases or close variants may appear in training.
- Those showcase cases are rehearsed demonstration cases, not evidence of generalization.
- Direct deterministic gold serialization is preferred over teacher generation.
- If linguistic target content is unavoidable, request one teacher candidate and keep it only after parser/validator acceptance.
- No rationale unless the protocol explicitly requires it.
- Keep prompt plus target near/under 1,024 tokens.
- Keep output near/under 128 tokens.
- Reject empty, unparsable, or invalid targets.
- Inspect 25 random rows plus all showcase cases.
- Freeze deterministic seeds and dataset manifest/hash.

Flash outer row transport:

```json
{"input":"<protocol-rendered input>","output":"<protocol-serialized target>","metadata":{"case_id":"<id>","seed":0,"stratum":"<protocol-owned label>"}}
```

This example specifies only Flash's outer envelope. Inner contents remain protocol-owned.

## Inference/demo contract

Call the deployed Flash run endpoint with:

- `temperature: 0.0`
- `max_tokens: 128` or smaller when protocol-safe
- 10–12 second total timeout
- one identical retry
- parse and deterministic validation before any state mutation

Response text is at `choices[0].message.content`.

Runtime modes:

1. `live` — call deployed adapter.
2. `cached-trained` — replay responses captured from the exact run/checkpoint ID.
3. `deterministic-replay` — use simulator/oracle fallback.

Runtime flow:

```text
request trained adapter
→ timeout?
    → retry once, then cache/fallback
→ parse failure?
    → retry once, then cache/fallback
→ validator rejection?
    → show rejection, then cache/fallback
→ accepted
    → apply deterministic state transition and animate
```

The application server owns the Freesolo API key; never expose it to browser JavaScript.

Demo appearance priorities:

- event card enters;
- trained-policy badge lights during inference;
- human-readable proposed action/audit trail;
- visible validator acceptance/rejection reason;
- 400–800 ms graph/state animation;
- stock-versus-tuned toggle using measured outputs;
- small footer with model, run/checkpoint ID, runtime mode, and latency;
- genuine trained responses cached before recording.

The presenter must be honest if the system fails over from live to cached/replay mode.

## Current local artifacts

### Created

- `/Users/kuanw/Me/code/Hackathon/Research/Small Model Training Pipeline Memory.md`
  - 574 lines.
  - Full literature, products, algorithms, gates, datasets, security, Mem0 analysis, and protocol boundary.
- `/Users/kuanw/Me/code/Hackathon/24-Hour One-Shot Training and Demo Runbook.md`
  - 378 lines.
  - Authoritative current execution plan.
- `/Users/kuanw/Me/code/Hackathon/train/configs/sft-one-shot.toml`
  - Valid TOML; one-shot invariants checked.
- `/Users/kuanw/Me/code/Hackathon/train/PROTOCOL_HANDOFF.md`
  - Minimal protocol-to-training contract checklist.
- `/Users/kuanw/Me/code/Hackathon/train/FLASH_RUN_LOG.md`
  - Credential-safe run receipt and demo probe table.
- `/Users/kuanw/Me/code/Hackathon/train/README.md`
  - Launch sequence and artifact guide.

### Updated

- `/Users/kuanw/Me/code/Hackathon/Project Plan.md`
  - Current MVP changed to one SFT deployment plus reliable demo; GRPO moved to stretch.
- `/Users/kuanw/Me/code/Hackathon/Fine-Tuning Plan.md`
  - Warning added that the one-shot runbook supersedes the research-heavy pipeline for the current deadline.
- `/Users/kuanw/Me/code/Hackathon/.gitignore`
  - Added `.env`, local Python artifacts, macOS files, and local Flash secret directories.

### Existing source material

- `/Users/kuanw/Me/code/Hackathon/.docs/Post Training Slides.pdf`
- `/Users/kuanw/Me/code/Hackathon/System Architecture.md`
- `/Users/kuanw/Me/code/Hackathon/General Idea.md`
- `/Users/kuanw/Me/code/Hackathon/Prize Strategy.md`
- `/Users/kuanw/Me/code/Hackathon/Research/Agent Memory Landscape.md`
- other belief-revision, calibration, graph, fraud, and prompt-injection research notes in `Research/`.

## Current machine and repository state

At export time:

- workspace: `/Users/kuanw/Me/code/Hackathon`
- thread ID: `019f72d4-201a-7e62-9553-fae764c9a201`
- date/timezone context: 2026-07-17, America/Toronto
- installed `python3`: `3.14.6`
- `uv`: not found
- `flash`: not found
- Flash 1.0.0 requires Python `>=3.11,<3.13`
- no Freesolo authentication was performed
- no environment was published
- no paid training was submitted
- no run ID/checkpoint exists
- no adapter was deployed
- no trained response cache exists

Git working tree at export time contains uncommitted changes:

```text
 M "Fine-Tuning Plan.md"
 M "Project Plan.md"
?? .gitignore
?? "24-Hour One-Shot Training and Demo Runbook.md"
?? "Research/Small Model Training Pipeline Memory.md"
?? train/
```

Do not discard or overwrite these changes. They are the output of this conversation and may coexist with user work.

## Immediate next actions

### Human/training operator

1. Obtain/export `FREESOLO_API_KEY` without writing it to the repository.
2. Install isolated tooling:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv tool install --python 3.12 'freesolo-flash==1.0.0'
flash version
flash login --api-key "$FREESOLO_API_KEY"
flash whoami
flash models
```

3. Escalate to Freesolo immediately if auth/model listing fails by Hour 1.

### Protocol/data owner

1. Complete `train/PROTOCOL_HANDOFF.md`.
2. Deliver the six minimal interfaces/fixtures by Hour 2.
3. Generate/freeze 256 accepted train rows, 32 canaries, five showcase cases, and hashes.
4. Verify every training target round-trips through the real parser/validator.

### Training operator after handoff

```bash
flash env push --name holdfast-demo train
# Replace the placeholder [environment].id in train/configs/sft-one-shot.toml.
cd train
flash train configs/sft-one-shot.toml --dry-run
flash train configs/sft-one-shot.toml --cost
flash train configs/sft-one-shot.toml --background
```

Record the run ID and all status/deployment data in `train/FLASH_RUN_LOG.md`.

### App/demo owner

1. Implement the Flash server-side client with timeout/retry.
2. Route output through parser and validator.
3. Implement live/cached-trained/deterministic-replay modes.
4. Freeze five showcase cases.
5. After deployment, capture genuine trained responses with run/checkpoint ID and latency.
6. Record the backup demonstration by Hour 18.

## Stop conditions

- No protocol adapter by H2: protocol owner freezes the smallest executable subset; training owner does not invent schema.
- Fewer than 128 valid examples by H4: train on only the valid examples and lower `max_examples`.
- Dry-run failure: fix only reported config/contract failures.
- Run still queued at H10: continue app against stub/replay; poll in background.
- Model format failure: one retry, then trained-cache or deterministic fallback.
- Unstable endpoint by H15: present cached-trained mode with terminal/run evidence.
- No deployment by H18: record deterministic demo plus Flash run/status evidence; recover during H24–36.

## Source/version notes

- Flash execution details were checked against [Freesolo Flash 1.0.0](https://pypi.org/project/freesolo-flash/1.0.0/), released 2026-07-16.
- The package documents managed LoRA SFT/GRPO/OPD, environment publishing, server dry-run, cost preflight, run monitoring, deployment, and chat-completions-shaped serving.
- The live installed CLI and `flash train --doc` are authoritative when implementation begins.
- Product behavior may change; pin package version, model revision where practical, environment ID, data hash, config, run ID, and deployed checkpoint.
- Research cutoff for the memory document is 2026-07-17.

## Recommended reading order in the next discussion

1. `CONVERSATION_CONTEXT.md` — this handoff.
2. `24-Hour One-Shot Training and Demo Runbook.md` — authoritative current execution.
3. `train/PROTOCOL_HANDOFF.md` — blocker owned by the other workstream.
4. `train/configs/sft-one-shot.toml` — one paid run.
5. `train/FLASH_RUN_LOG.md` — execution receipt.
6. `Research/Small Model Training Pipeline Memory.md` — full research-quality pipeline and literature.
7. `Project Plan.md` and `Fine-Tuning Plan.md` — broader/staged plans, now overridden for the hackathon.

## Do not lose these decisions

- The demo deadline, not accuracy, controls the current implementation.
- One paid SFT run only.
- Qwen3.5-4B is the student for this run.
- No GRPO or OPD before the demo is ready.
- The shared protocol workstream owns all domain serialization.
- The deterministic simulator/validator is authoritative.
- The LLM only proposes state changes.
- Cache only real trained responses.
- Maintain an honest replay fallback.
- Never expose the API key in frontend code, dataset files, Markdown, screenshots, git, or recordings.
- Brier scoring needs a fixed proposition universe if RL is added later.
- Mem0 is an inference-time baseline, not the training recipe.
