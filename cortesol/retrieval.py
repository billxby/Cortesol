"""Retrieval — build the bounded Context the extractor reads (area C).

Embed the incoming event -> top-k relevant claims + their 1-hop neighborhood +
source records (config.RETRIEVE_TOP_K). Must fit CONTEXT_TOKEN_BUDGET. Both the
live pipeline and the GRPO environment call this, so keep it dependency-light; a
trivial tag/recency version works before embeddings land.

Reference: System Architecture §lifecycle step 2.
"""

from __future__ import annotations

from .core.context import Context
from .core.kb import KB
from .core.schema import Evidence, RawEvent


def retrieve(kb: KB, event: RawEvent, evidence: Evidence, k: int | None = None) -> Context:
    """Select the top-k relevant claims + neighborhood and pack a Context."""
    ...  # TODO
