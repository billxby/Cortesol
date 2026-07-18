# Cortesol — common commands. Run `make help` for the list.
# These names are a contract: CLAUDE.md and the branch briefs reference them.

.PHONY: help install test test-contract test-unit test-integration lint fmt \
        sim fetch-papers bootstrap eval run-ui run-ui-foundation train-sft train-grpo \
        training-preflight snapshot-clean plan-b-setup plan-b-serve plan-b-tunnel

CONDA_ENV := cortesol-train
RUN := conda run --no-capture-output -n $(CONDA_ENV)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Create/update the authoritative Conda environment
	conda env update -n $(CONDA_ENV) -f environment.yml --prune

test:  ## Run the whole suite
	$(RUN) python -m pytest

test-contract:  ## Run ONLY contract tests — the merge gate. Must be green on every branch.
	$(RUN) python -m pytest -m contract

test-unit:  ## Run unit tests
	$(RUN) python -m pytest -m unit

test-integration:  ## Run cross-branch integration tests
	$(RUN) python -m pytest -m integration

lint:  ## Static checks
	$(RUN) python -m ruff check cortesol tests

fmt:  ## Auto-format
	$(RUN) python -m ruff format cortesol tests
	$(RUN) python -m ruff check --fix cortesol tests

sim:  ## [B2] Generate a toy peptide stream from the simulator and print it
	$(RUN) python -m cortesol.sim.world --demo

fetch-papers:  ## [B/C] Fetch real peptide abstracts from PubMed -> data/papers/ (evidence, no oracle)
	$(RUN) python -m cortesol.ingest.fetch_papers --per-peptide 8

bootstrap:  ## [C] Build the demo foundation: replay 3 critical papers/peptide through the engine -> data/snapshots/
	$(RUN) python -m cortesol.bootstrap --per-peptide 3

eval:  ## [B3] Replay held-out streams and print the metrics table
	$(RUN) python -m cortesol.eval.replay

run-ui:  ## [B3] Launch the live belief-graph demo (FastAPI + SSE)
	$(RUN) uvicorn cortesol.ui.app:app --reload --port 8000

run-ui-foundation:  ## [C] Launch the demo pre-seeded with the foundation (run `make bootstrap` first)
	CORTESOL_FOUNDATION=1 $(RUN) uvicorn cortesol.ui.app:app --reload --port 8000

plan-b-setup:  ## Pull Qwen 3.5 9B and build the schema-policy Ollama alias
	ollama pull qwen3.5:9b
	ollama create cortesol-proposer:plan-b -f deploy/ollama/Modelfile

plan-b-serve:  ## Serve the authenticated Freesolo-compatible Plan B gateway
	$(RUN) uvicorn cortesol.serve.gateway:app --host 127.0.0.1 --port 8787

plan-b-tunnel:  ## Expose the Plan B gateway through a Cloudflare quick tunnel
	cloudflared tunnel --url http://127.0.0.1:8787 --http-host-header 127.0.0.1:8787

train-sft:  ## [B2] Build the SFT dataset and launch the Flash SFT run
	$(RUN) python -m cortesol.train.coordinator run --from-stage smoke_sft

train-grpo:  ## [B2] Launch the Flash GRPO run against the real engine
	$(RUN) python -m cortesol.train.coordinator run --from-stage primary_grpo

training-preflight:  ## Build, test, dry-run, estimate cost, and stop for approval
	$(RUN) python -m cortesol.train.coordinator preflight

snapshot-clean:  ## Wipe local KB snapshots
	rm -rf snapshots/*.json
