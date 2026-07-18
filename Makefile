# Cortesol — common commands. Run `make help` for the list.
# These names are a contract: CLAUDE.md and the branch briefs reference them.

.PHONY: help install test test-contract test-unit test-integration lint fmt \
        sim fetch-papers eval run-ui train-sft train-grpo snapshot-clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Create the env and install deps (uv)
	uv sync --extra dev

test:  ## Run the whole suite
	uv run pytest

test-contract:  ## Run ONLY contract tests — the merge gate. Must be green on every branch.
	uv run pytest -m contract

test-unit:  ## Run unit tests
	uv run pytest -m unit

test-integration:  ## Run cross-branch integration tests
	uv run pytest -m integration

lint:  ## Static checks
	uv run ruff check cortesol tests

fmt:  ## Auto-format
	uv run ruff format cortesol tests
	uv run ruff check --fix cortesol tests

sim:  ## [B2] Generate a toy peptide stream from the simulator and print it
	uv run python -m cortesol.sim.world --demo

fetch-papers:  ## [B/C] Fetch real peptide abstracts from PubMed -> data/papers/ (evidence, no oracle)
	uv run python -m cortesol.ingest.fetch_papers --per-peptide 8

eval:  ## [B3] Replay held-out streams and print the metrics table
	uv run python -m cortesol.eval.replay

run-ui:  ## [B3] Launch the live belief-graph demo (FastAPI + SSE)
	uv run uvicorn cortesol.ui.app:app --reload --port 8000

train-sft:  ## [B2] Build the SFT dataset and launch the Flash SFT run
	uv run python -m cortesol.train.make_sft && flash train cortesol/train/configs/sft.toml

train-grpo:  ## [B2] Launch the Flash GRPO run against the real engine
	flash train cortesol/train/configs/grpo.toml

snapshot-clean:  ## Wipe local KB snapshots
	rm -rf snapshots/*.json
