"""The deterministic validator — the gatekeeper. THE GUARANTEE.

OWNER: Branch 1 (ground-truth). This is the single write path into belief state:
the orchestrator submits ProposedOps, the validator returns what is allowed, and
ONLY accepted ops reach the engine. Even a fully compromised extractor cannot get
a forbidden change past this file. (Prompt Injection Defense §Layer-4, PD1/PD2/PD4.)

This is a STUB. Every check below is deterministic and cheap. Implement, then make
tests/unit/test_validator.py and the contract tests pass.

The invariants (Prompt Injection Defense §Layer-4, System Architecture §validate):
  1. Enumerated ops only — the type system already guarantees this (ops.Op). A
     malformed op never parses; treat parse failure as REJECT(malformed).
  2. Bounded step — no APPLY_EVIDENCE may move a claim by more than DELTA_MAX in
     |Δell| after caps/damping. (The engine computes the move; the validator sets
     the ceiling and rejects ops that would require exceeding it.)
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

from . import config, domain
from .kb import KB
from .ops import AddClaim, AddEdge, ApplyEvidence, FlagOOD, InvalidateEdge, ProposedOps, Reject
from .results import RejectedOp, ValidationResult
from .schema import ClaimStatus, Evidence


def _tags_in_scope(tags: list[str]) -> bool:
    namespaces = {tag.split(":", 1)[0] for tag in tags if ":" in tag}
    return (
        "peptide" in namespaces
        and bool(namespaces & set(domain.PROPERTY_TYPES))
        and namespaces <= set(domain.IN_SCOPE_NAMESPACES)
    )


def _conflict_mass(kb: KB, op: ApplyEvidence) -> float:
    claim = kb.get_claim(op.claim_id)
    if claim is None:
        return 0.0
    incoming = {"weak": 0.25, "moderate": 0.6, "strong": 0.9}[op.strength]
    opposing = claim.opinion["d"] if op.direction == "+" else claim.opinion["b"]
    return float(incoming * opposing)


def validate(kb: KB, proposed: ProposedOps, evidence: Evidence) -> ValidationResult:
    """Screen ProposedOps against the invariants above. Returns accepted +
    rejected(op, reason). The orchestrator commits only `result.accepted`.

    NOTE: `proposed.think` is IGNORED here — rationale never authorizes an action.
    Only `proposed.ops` are considered. (PD3.)
    """
    accepted = []
    rejected: list[RejectedOp] = []
    attributed = 0

    for op in proposed.ops:
        reason: str | None = None
        if isinstance(op, (FlagOOD, Reject)):
            accepted.append(op)
            continue

        attributed += 1
        if {"prompt_injection", "out_of_scope"} & set(evidence.red_flags):
            reason = "state-changing operations are forbidden for unsafe or out-of-scope evidence"
        elif attributed > config.MAX_OPS_PER_SOURCE_PER_EVENT:
            reason = "per-source operation rate limit exceeded"
        elif isinstance(op, ApplyEvidence):
            claim = kb.get_claim(op.claim_id)
            if claim is None:
                reason = f"unknown claim {op.claim_id!r}"
            elif claim.status in {ClaimStatus.QUARANTINED, ClaimStatus.RETIRED}:
                reason = f"claim {op.claim_id!r} is not writable"
            elif op.evidence_id != evidence.id:
                reason = "missing or mismatched evidence provenance"
            elif kb.get_source(evidence.source_id) is None:
                reason = f"unknown source {evidence.source_id!r}"
            elif _conflict_mass(kb, op) > config.CONFLICT_MASS_THRESHOLD:
                claim.status = ClaimStatus.QUARANTINED
                reason = "high Dempster-Shafer conflict; claim quarantined"
        elif isinstance(op, AddClaim):
            if not op.text.strip():
                reason = "claim text is empty"
            elif not _tags_in_scope(op.ontology_tags):
                reason = "claim ontology tags are out of scope"
            elif op.initial_evidence_id not in {None, evidence.id}:
                reason = "mismatched initial evidence provenance"
        elif isinstance(op, AddEdge):
            if op.src == op.dst:
                reason = "self-loop edges are not allowed"
            elif op.src not in kb.claims or op.dst not in kb.claims:
                reason = "edge endpoint does not exist"
            elif not math.isfinite(op.weight) or not 0.0 <= op.weight <= 1.0:
                reason = "edge weight must be finite and in [0, 1]"
        elif isinstance(op, InvalidateEdge):
            edge = kb.edges.get(op.edge_id)
            if edge is None or not edge.live:
                reason = "edge does not exist or is already invalid"
            elif op.evidence_id != evidence.id:
                reason = "missing or mismatched invalidation provenance"

        if reason:
            rejected.append(RejectedOp(op=op, reason=reason))
        else:
            accepted.append(op)

    return ValidationResult(accepted=accepted, rejected=rejected)
