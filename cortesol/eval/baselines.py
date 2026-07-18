"""Baselines (area C) — the anchors that make the numbers mean something.

gullible bot (apply everything), stubborn bot (ignore everything) -> the two
CORTEX failure modes ("the entire challenge is doing neither"). Plus raw frontier
LLM with the KB in-context, and engine+{stock, SFT, GRPO} -> the Freesolo story
(training moved every metric). Each is a drop-in extractor the pipeline runs.

Each baseline exposes the same duck-typed interface as `ingest.extract.FakeExtractor`
and the real tuned extractor: `extract(ctx, model=None) -> ProposedOps`. So they
drop straight into `pipeline.process_event(kb, event, extractor=...)`. Crucially,
even the gullible bot cannot escape the validator: its strongest possible output is
one capped APPLY_EVIDENCE — the DELTA_MAX bound holds regardless of the extractor
(Prompt Injection Defense §Layer-4). That is exactly what these anchors demonstrate.

Three of the four model-backed baselines (engine+stock, engine+SFT, engine+GRPO)
are the SAME code path — the product architecture (engine behind the validator)
with a different extractor checkpoint — so they share one `ModelBaseline` wrapper
that delegates to `ingest.extract.extract`. They need a model/network, so they
belong behind the `slow` test marker, never in the offline `make eval` default.
The fourth, "raw frontier LLM with the KB in-context", is architecturally
different (it asks the LLM to hold beliefs directly, with NO engine) — it is an
ablation path, not a drop-in extractor, and is noted at the bottom of this file.
"""

from __future__ import annotations

import os

from ..core.context import Context
from ..core.ops import ApplyEvidence, ProposedOps


class GullibleBot:
    """Applies every result at full strength — believes whatever it was told last.

    For each event it emits a single strong, positive APPLY_EVIDENCE against the
    most-relevant retrieved claim, with no skepticism, no screening, no direction
    reasoning. It over-amplifies genuine, hyped, and fraudulent evidence alike ->
    high Brier / high ECE. (The validator still caps each move, which is the point:
    high miscalibration, but never an unbounded jump.)
    """

    name = "gullible"

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        if not ctx.claims:
            # Nothing retrieved to move; a gullible bot introduces no structure.
            return ProposedOps(think="gullible: no known claim to amplify")
        target = ctx.claims[0]  # top-k is relevance-ordered
        op = ApplyEvidence(
            claim_id=target.id,
            direction="+",
            strength="strong",
            evidence_id=ctx.evidence.id,
        )
        return ProposedOps(think="gullible: believe it, fully", ops=[op])


class StubbornBot:
    """Emits no belief-moving ops — ignores everything.

    The opposite failure mode: perfectly injection-proof and fraud-proof (ASR and
    fraud-rate 0), but it never learns, so every claim stays pinned at the prior
    and it can never revise toward a real, well-replicated result."""

    name = "stubborn"

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        return ProposedOps(think="stubborn: ignore everything")


class ModelBaseline:
    """Engine + an LLM extractor — the product architecture, parameterized by
    checkpoint. Only the model changes across the Freesolo ablation:

        engine+stock -> untuned base 4B     (FLASH_MODEL_STOCK)
        engine+SFT   -> after supervised fine-tune
        engine+GRPO  -> after GRPO, the shipped policy (FLASH_MODEL_TUNED)

    It delegates to `ingest.extract.extract`, so it goes live the moment that lands
    (today `extract` is a stub). Network/model-gated: run it behind the `slow` test
    marker, not in the offline `make eval` default.
    """

    def __init__(self, name: str, model: str | None = None) -> None:
        self.name = name
        self.model = model

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        # Lazy import so merely importing baselines never constructs an API client.
        from ..ingest.extract import extract as _extract

        return _extract(ctx, model=self.model or model)


def engine_stock() -> ModelBaseline:
    """engine + the untuned base model (FLASH_MODEL_STOCK from .env)."""
    return ModelBaseline("engine+stock", os.getenv("FLASH_MODEL_STOCK"))


def engine_tuned(name: str = "engine+grpo", model: str | None = None) -> ModelBaseline:
    """engine + a tuned adapter. SFT and GRPO are the same wrapper with different
    adapter ids; defaults to FLASH_MODEL_TUNED (the shipped GRPO policy)."""
    return ModelBaseline(name, model or os.getenv("FLASH_MODEL_TUNED"))


# The "raw frontier LLM with the KB in-context" baseline is deliberately NOT here.
# It bypasses the engine — the LLM is asked to output confidences directly — so it
# does not fit the `extract() -> ProposedOps` interface the validator/engine expect.
# It is the ablation that demonstrates uncalibrated, ledger-free belief; it belongs
# as a separate no-engine path in `replay.py`, driven by OPENAI_API_KEY/TEACHER_MODEL.
