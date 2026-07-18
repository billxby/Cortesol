"""Baselines (area C) — the anchors that make the numbers mean something.

gullible bot (apply everything), stubborn bot (ignore everything) -> the two
CORTEX failure modes ("the entire challenge is doing neither"). Plus raw frontier
LLM with the KB in-context, and engine+{stock, SFT, GRPO} -> the Freesolo story
(training moved every metric). Each is a drop-in extractor the pipeline runs.
"""

from __future__ import annotations

from ..core.context import Context
from ..core.ops import ProposedOps


class GullibleBot:
    """Applies every result at full strength — believes whatever it was told last."""

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        ...  # TODO


class StubbornBot:
    """Emits no belief-moving ops — ignores everything."""

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        ...  # TODO
