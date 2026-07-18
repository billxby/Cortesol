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

from .kb import KB
from .ops import ProposedOps
from .results import ValidationResult
from .schema import Evidence


def validate(kb: KB, proposed: ProposedOps, evidence: Evidence) -> ValidationResult:
    """Screen ProposedOps against the invariants above. Returns accepted +
    rejected(op, reason). The orchestrator commits only `result.accepted`.

    NOTE: `proposed.think` is IGNORED here — rationale never authorizes an action.
    Only `proposed.ops` are considered. (PD3.)
    """
    raise NotImplementedError("BRANCH-1: implement the six invariant checks")
