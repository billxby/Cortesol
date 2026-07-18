# Cortesol — common commands. Run `make help` for the list.
# These names are a contract: CLAUDE.md and the branch briefs reference them.

.PHONY: help install test test-contract test-unit test-integration lint fmt \
        sim eval run-ui train-sft train-grpo training-preflight snapshot-clean

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

eval:  ## [B3] Replay held-out streams and print the metrics table
	$(RUN) python -m cortesol.eval.replay

run-ui:  ## [B3] Launch the live belief-graph demo (FastAPI + SSE)
	$(RUN) uvicorn cortesol.ui.app:app --reload --port 8000

train-sft:  ## [B2] Build the SFT dataset and launch the Flash SFT run
	$(RUN) python -m cortesol.train.coordinator run --from-stage smoke_sft

train-grpo:  ## [B2] Launch the Flash GRPO run against the real engine
	$(RUN) python -m cortesol.train.coordinator run --from-stage primary_grpo

training-preflight:  ## Build, test, dry-run, estimate cost, and stop for approval
	$(RUN) python -m cortesol.train.coordinator preflight

snapshot-clean:  ## Wipe local KB snapshots
	rm -rf snapshots/*.json
