"""Deterministic gold ops — the labels, computed not judged (area B).

gold_op = pure_function(world_state, event_class). No LLM, no labeling. This is
the APIGen lesson: keep only what a deterministic verifier certifies. The gold
lives in RawEvent.sim_meta and is used for SFT targets, GRPO reward, and eval —
never shown to the extractor (PD6).

Reference: Fine-Tuning Plan §Stage 0.
  genuine+strong -> APPLY_EVIDENCE(+, strong); hyped -> damped + flag;
  fraudulent/injection -> REJECT(reason); out_of_scope -> FLAG_OOD.
"""

from __future__ import annotations

from ..core.ops import ProposedOps
from ..core.schema import RawEvent


def gold_ops(event: RawEvent) -> ProposedOps:
    """Return the gold ProposedOps for an event from its sim_meta (world + class)."""
    ...  # TODO
