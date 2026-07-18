# Full gated training pipeline

The coordinator owns the lineage:

```text
smoke SFT -> production SFT -> primary GRPO
                         \-> managed GLM-5.2 OPD -> GRPO
                         -> sealed comparison -> deployed winner
```

## One-time setup

```bash
conda env update -n cortesol-train -f environment.yml --prune
conda activate cortesol-train
export FREESOLO_API_KEY=fslo_...
```

`FIREWORKS_API_KEY` and `OPENAI_API_KEY` are not used. Flash supplies the managed
OPD teacher credential. Keep the Freesolo key in the shell or `.env.local`; never
put it in a TOML file or coordinator state.

## Preflight and approval

```bash
make training-preflight
```

Preflight fetches `origin/main` and `feature/training`, verifies the branch,
runs the full test/lint suite, regenerates and hashes all datasets, builds the
publishable bundle without development/final/security data, publishes the
environment, parses every config with Flash 1.0.1, performs the immediately
resolvable server dry-runs, and writes the combined training plus worst-case
evaluation-token estimate to `runs/training/state.json`.

It then stops. Approve a cap at or above the printed estimate:

```bash
python -m cortesol.train.coordinator approve --cap-usd 60
python -m cortesol.train.coordinator run
```

Every future warm-start is server dry-run again after its parent checkpoint
exists and before submission. If a refreshed server quote would put the lineage
above the approved cap, the coordinator stops without submitting that run.

The coordinator is resumable: rerun the last command after an interruption. It
records environment/run/checkpoint IDs, costs, metrics, gate failures, the sealed
evaluation latch, and deployment state, but never credentials.

Flash 1.0.1 enables contexts up to 32,768 tokens for models at or below 9B. The
generated 24-turn Cortesol transcripts top out at an estimated 7,497 tokens, so
the 4B GRPO and OPD runs use 12,288 tokens: enough measured headroom without
paying the GPU-memory cost of an unused 32k allocation. OPD pins `group_size=1`,
which Flash recommends for distillation and which keeps the resident trainer and
rollout engine within a validated GPU. Deterministic-gold SFT remains at 2,048.

## Teacher-filtered SFT lineage

Build the deterministic source splits first, then sample candidates from an
immutable deployed Freesolo adapter. Gold operations are used only by the local
filter and never appear in the teacher prompt:

```bash
python -m cortesol.train.make_sft --out runs/training/source
python -m cortesol.train.teacher_filter \
  --source runs/training/source \
  --out runs/training/data \
  --cache runs/training/teacher-cache \
  --teacher-revision 'RUN_ID@IMMUTABLE_REVISION' \
  --k 4
python -m cortesol.train.coordinator preflight \
  --environment-name cortesol-teacher-rft \
  --reuse-prepared-data
```

The filter requires schema-valid operations that exactly match simulator gold
and a 3–32 word rationale. Candidate responses are append-only cached so an
interrupted generation resumes without paying for completed calls. Prepared-data
preflight re-hashes every split before publishing it to Freesolo.
