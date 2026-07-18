"""The extractor — the LLM update policy (logical area B, the Freesolo model).

Reads a serialized Context (quarantined event + retrieved state) and emits
ProposedOps under JSON-schema-constrained decoding (ops.ops_json_schema()).
Points the openai client at the Flash deployment (.env FLASH_*). This is the model
we fine-tune: stock 4B -> SFT -> GRPO -> OPD.

Reference: System Architecture §lifecycle step 3, Fine-Tuning Plan.
The model PROPOSES; it never writes state. Its `think` trace is advisory only.
"""

from __future__ import annotations

from ..core.context import Context
from ..core.ops import ProposedOps


def extract(ctx: Context, model: str | None = None) -> ProposedOps:
    """Call the (tuned) model with the op schema and return ProposedOps."""
    ...  # TODO


class FakeExtractor:
    """Canned proposer so areas A and C can run the pipeline with no model/network.
    Return scripted ProposedOps keyed by event id (load from a fixture)."""

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        ...  # TODO
