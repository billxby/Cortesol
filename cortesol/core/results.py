"""Interface result types — the language the components speak to each other.

FROZEN CONTRACT (steward: Branch 1). These are the return types that cross branch
boundaries: the validator returns a `ValidationResult`, the engine and propagate
return `Delta`s, and the orchestrator (Branch 3) assembles an `EventResult` for
the audit log, the SSE stream, and the eval harness. Changing a field here ripples
into all three branches, so it is a contract amendment.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .ops import Op


class Delta(BaseModel):
    """One belief change, fully attributed for the audit log. Every Δell in the
    system has one of these — 'every Δℓ has a cause chain' is the anti-black-box
    pitch (System Architecture §why-this-is-the-pitch)."""

    claim_id: str
    before_ell: float
    after_ell: float
    cause: str  # human-readable: which op / propagation caused it
    op: str | None = None  # op name, if a direct op (vs. a propagation ripple)


class RejectedOp(BaseModel):
    op: Op
    reason: str  # which invariant it violated


class ValidationResult(BaseModel):
    """The validator's verdict on one event's ProposedOps."""

    accepted: list[Op] = Field(default_factory=list)
    rejected: list[RejectedOp] = Field(default_factory=list)

    @property
    def all_rejected(self) -> bool:
        return not self.accepted and bool(self.rejected)


class AuditEntry(BaseModel):
    """One line in the immutable audit log. Injections, rejects, and every belief
    move land here — this is what a judge clicks to see *why* a node moved."""

    t: int
    kind: str  # "commit" | "reject" | "flag_ood" | "propagate" | "discredit"
    detail: str
    op: str | None = None
    delta: Delta | None = None


class EventResult(BaseModel):
    """The full outcome of processing one event — the orchestrator's return value
    and the unit the eval harness and UI consume."""

    event_id: str
    t: int
    validation: ValidationResult
    deltas: list[Delta] = Field(default_factory=list)  # direct + propagated
    audit: list[AuditEntry] = Field(default_factory=list)
    dirty_claims: list[str] = Field(default_factory=list)  # for the UI pulse
