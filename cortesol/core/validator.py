"""The deterministic validator — the gatekeeper. THE GUARANTEE.

OWNER: Branch 1 (ground-truth). This is the single write path into belief state:
the orchestrator submits ProposedOps, the validator returns what is allowed, and
ONLY accepted ops reach the engine. Even a fully compromised extractor cannot get
a forbidden change past this file. (Prompt Injection Defense §Layer-4, PD1/PD2/PD4.)

The invariants (Prompt Injection Defense §Layer-4, System Architecture §validate):
  1. Enumerated ops only — the type system already guarantees this (ops.Op). A
     malformed op never parses; treat parse failure as REJECT(malformed).
  2. Bounded step — no APPLY_EVIDENCE may move a claim by more than DELTA_MAX in
     |Δell| after caps/damping. The engine computes the move (always clipped to
     DELTA_MAX); the validator refuses ops whose provenance can't justify a move.
  3. Provenance required — APPLY_EVIDENCE / INVALIDATE_EDGE must cite an
     evidence_id whose source exists and passed screening.
  4. Referential integrity — claim_id / edge_id / endpoints must exist;
     ADD_CLAIM tags must sit in the domain ontology else it should have been FLAG_OOD.
  5. Rate limits — at most MAX_OPS_PER_SOURCE_PER_EVENT ops attributed to one
     source in one event.
  6. Conflict quarantine — if incoming evidence conflicts hard with current belief
     (DS conflict mass K > CONFLICT_MASS_THRESHOLD), do not fuse; quarantine.
FLAG_OOD and REJECT are ALWAYS allowed (flagging/refusing is safe).
"""

from __future__ import annotations

import math

from .config import (
    CONFLICT_MASS_THRESHOLD,
    MAX_OPS_PER_SOURCE_PER_EVENT,
)
from .domains import get_active_domain
from .kb import KB
from .ops import (
    AddClaim,
    AddEdge,
    ApplyEvidence,
    FlagOOD,
    InvalidateEdge,
    ProposedOps,
    Reject,
)
from .results import RejectedOp, ValidationResult
from .schema import ClaimStatus, Evidence


def _tag_in_ontology(tag: str) -> bool:
    namespace = tag.split(":", 1)[0]
    return namespace in get_active_domain().in_scope_namespaces


def _hard_conflict(kb: KB, op: ApplyEvidence) -> bool:
    """Dempster-Shafer conflict mass between current belief and the incoming
    report's direction. High only when a confident belief meets an equally
    confident *opposite* report — that is the case we quarantine rather than fuse.
    Ordinary revision (a contradiction against a still-uncertain claim) passes, so
    the system stays revisable, not stubborn."""
    claim = kb.get_claim(op.claim_id)
    if claim is None:
        return False
    incoming = {"weak": 0.25, "moderate": 0.6, "strong": 0.9}[op.strength]
    opposing = claim.opinion["d"] if op.direction == "+" else claim.opinion["b"]
    return incoming * opposing > CONFLICT_MASS_THRESHOLD


def validate(kb: KB, proposed: ProposedOps, evidence: Evidence) -> ValidationResult:
    """Screen ProposedOps against the invariants above. Returns accepted +
    rejected(op, reason). The orchestrator commits only `result.accepted`.

    NOTE: `proposed.think` is IGNORED here — rationale never authorizes an action.
    Only `proposed.ops` are considered. (PD3.)
    """
    result = ValidationResult()
    source = kb.get_source(evidence.source_id)
    provenance_ok = source is not None and evidence.id  # source exists & screened

    attributed = 0  # ops charged against this event's source (rate limit)

    for op in proposed.ops:
        # (1) enumerated ops only — guaranteed by parsing into the Op union. FLAG_OOD
        # and REJECT are always safe: flagging/refusing never mutates belief.
        if isinstance(op, (FlagOOD, Reject)):
            result.accepted.append(op)
            continue

        # (5) per-source rate limit on state-changing ops
        if attributed >= MAX_OPS_PER_SOURCE_PER_EVENT:
            result.rejected.append(RejectedOp(op=op, reason="rate_limit_exceeded"))
            continue

        if {"prompt_injection", "out_of_scope"} & set(evidence.red_flags):
            result.rejected.append(
                RejectedOp(
                    op=op,
                    reason="state-changing operations forbidden for unsafe evidence",
                )
            )
            continue

        if isinstance(op, ApplyEvidence):
            # (3) provenance
            if not provenance_ok or op.evidence_id != evidence.id:
                result.rejected.append(RejectedOp(op=op, reason="missing_or_unscreened_provenance"))
                continue
            # (4) referential integrity
            if kb.get_claim(op.claim_id) is None:
                result.rejected.append(RejectedOp(op=op, reason="unknown_claim"))
                continue
            claim = kb.get_claim(op.claim_id)
            if claim is not None and claim.status in {
                ClaimStatus.QUARANTINED,
                ClaimStatus.RETIRED,
            }:
                result.rejected.append(RejectedOp(op=op, reason="claim_not_writable"))
                continue
            # (6) hard-conflict quarantine
            if _hard_conflict(kb, op):
                claim = kb.get_claim(op.claim_id)
                if claim is not None:
                    claim.status = ClaimStatus.QUARANTINED
                result.rejected.append(RejectedOp(op=op, reason="conflict_quarantine"))
                continue
            # (2) bounded step is guaranteed downstream by the engine's clip to
            #     DELTA_MAX; nothing here can request more.
            attributed += 1
            result.accepted.append(op)
            continue

        if isinstance(op, AddClaim):
            # (4) tags must sit within the ontology, else it should have been FLAG_OOD
            if not op.ontology_tags or not all(_tag_in_ontology(t) for t in op.ontology_tags):
                result.rejected.append(RejectedOp(op=op, reason="out_of_ontology_should_flag_ood"))
                continue
            if not op.text.strip():
                result.rejected.append(RejectedOp(op=op, reason="empty_claim_text"))
                continue
            if op.initial_evidence_id not in {None, evidence.id}:
                result.rejected.append(
                    RejectedOp(op=op, reason="mismatched_initial_provenance")
                )
                continue
            attributed += 1
            result.accepted.append(op)
            continue

        if isinstance(op, AddEdge):
            # (4) endpoints exist, no self-loop
            if op.src == op.dst:
                result.rejected.append(RejectedOp(op=op, reason="self_loop"))
                continue
            if kb.get_claim(op.src) is None or kb.get_claim(op.dst) is None:
                result.rejected.append(RejectedOp(op=op, reason="unknown_endpoint"))
                continue
            if not math.isfinite(op.weight) or not 0.0 <= op.weight <= 1.0:
                result.rejected.append(RejectedOp(op=op, reason="invalid_edge_weight"))
                continue
            attributed += 1
            result.accepted.append(op)
            continue

        if isinstance(op, InvalidateEdge):
            # (3) provenance + (4) referential integrity
            if not provenance_ok or op.evidence_id != evidence.id:
                result.rejected.append(RejectedOp(op=op, reason="missing_or_unscreened_provenance"))
                continue
            if op.edge_id not in kb.edges:
                result.rejected.append(RejectedOp(op=op, reason="unknown_edge"))
                continue
            if not kb.edges[op.edge_id].live:
                result.rejected.append(RejectedOp(op=op, reason="edge_already_invalid"))
                continue
            attributed += 1
            result.accepted.append(op)
            continue

        # unreachable given the closed Op union, but fail safe
        result.rejected.append(RejectedOp(op=op, reason="malformed"))

    return result
