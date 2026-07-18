"""Deterministic gold ops — the labels, computed not judged (area B).

gold_op = pure_function(world_state, event_class). No LLM, no labeling. This is
the APIGen lesson: keep only what a deterministic verifier certifies. The gold
lives in RawEvent.sim_meta and is used for SFT targets, GRPO reward, and eval —
never shown to the extractor (PD6).

Reference: Fine-Tuning Plan §Stage 0.
  genuine+strong -> APPLY_EVIDENCE(+, strong); hyped -> damped + flag;
  fraudulent/injection -> REJECT(reason); out_of_scope -> FLAG_OOD.

`build_gold` is the ONE place the class->op mapping lives; `sim/events.py` calls
it to populate `sim_meta.gold_ops`, so the stored gold and `gold_ops(event)`
below can never disagree. Gold op dicts are shaped exactly like `core.ops` models
(validated by `ProposedOps(ops=...)`).
"""

from __future__ import annotations

from typing import Any

from ..core.ops import ProposedOps
from ..core.schema import Direction, EventClass, RawEvent, Strength


def apply_evidence_gold(
    claim_id: str, direction: Direction, strength: Strength, evidence_id: str
) -> dict[str, Any]:
    return {
        "op": "APPLY_EVIDENCE",
        "claim_id": claim_id,
        "direction": direction,
        "strength": strength,
        "evidence_id": evidence_id,
    }


def reject_gold(evidence_id: str, reason: str) -> dict[str, Any]:
    return {"op": "REJECT", "evidence_id": evidence_id, "reason": reason}


def flag_ood_gold(payload: str, reason: str) -> dict[str, Any]:
    return {"op": "FLAG_OOD", "payload": payload, "reason": reason}


def build_gold(
    event_class: EventClass,
    *,
    evidence_id: str,
    claim_id: str | None = None,
    z: int | None = None,
    payload: str | None = None,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    """The class -> gold-op mapping. Pure function of (class, world-state).

    `z` is the latent truth of the referenced claim (1 true, 0 false); the sign
    of a NOISY update follows it. Returns raw op dicts to store in
    `sim_meta.gold_ops`.
    """
    if event_class is EventClass.GENUINE:
        assert claim_id is not None
        return [apply_evidence_gold(claim_id, "+", "strong", evidence_id)]

    if event_class is EventClass.NOISY:
        assert claim_id is not None
        direction: Direction = "+" if z else "-"
        return [apply_evidence_gold(claim_id, direction, "moderate", evidence_id)]

    if event_class is EventClass.HYPED:
        # Splashy but false (z=0): the gold op is a *weak* positive. The damping
        # and red-flags come from the engine/screen, not from the label.
        assert claim_id is not None
        return [apply_evidence_gold(claim_id, "+", "weak", evidence_id)]

    if event_class is EventClass.CONTRADICTORY:
        # Independent lab fails to replicate a true claim -> moderate negative.
        assert claim_id is not None
        return [apply_evidence_gold(claim_id, "-", "moderate", evidence_id)]

    if event_class is EventClass.FRAUDULENT:
        return [reject_gold(evidence_id, "unverifiable")]

    if event_class is EventClass.INJECTION:
        return [reject_gold(evidence_id, "injection")]

    if event_class is EventClass.OUT_OF_SCOPE:
        assert payload is not None
        return [flag_ood_gold(payload, reason or "outside the peptide ontology")]

    raise ValueError(f"unknown event class: {event_class!r}")


def gold_ops(event: RawEvent) -> ProposedOps:
    """Return the gold ProposedOps for an event from its sim_meta (world + class).

    Real (non-sim) events carry no gold; return an empty ProposedOps for them.
    """
    if event.sim_meta is None:
        return ProposedOps()
    return ProposedOps(ops=list(event.sim_meta.gold_ops))
