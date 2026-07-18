"""The op vocabulary — the closed set of state changes. THIS IS THE SECURITY
BOUNDARY.

FROZEN CONTRACT (steward: Branch 1; amendments require all-branch sign-off —
docs/PROTOCOL.md §Contract Freeze). The extractor (Branch 2) may ONLY emit these
ops; the validator (Branch 1) enforces them; the simulator's gold (Branch 2) is
expressed in them; the GRPO structured-outputs schema IS `ops_json_schema()`.

Notable absences are deliberate and load-bearing:
  * no SET_CONFIDENCE — you cannot assign belief, only supply evidence (PD2)
  * no DELETE — nothing is destroyed, only invalidated / retired (PD7)
  * no free-text field reaches storage as an instruction (PD3)

"Never let text rewrite what it knows" is enforced by this type, not by asking
the model nicely. A fully compromised LLM still cannot express a forbidden action
because there is no symbol for it. (Prompt Injection Defense §Layer-3/4.)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .schema import Direction, EdgeType, Strength

RejectReason = Literal["injection", "malformed", "unverifiable"]


class ApplyEvidence(BaseModel):
    """Move a claim's belief by one capped, provenance-carrying increment."""

    op: Literal["APPLY_EVIDENCE"] = "APPLY_EVIDENCE"
    claim_id: str
    direction: Direction  # "+" supports, "-" contradicts
    strength: Strength  # weak | moderate | strong (maps to log-Lambda pre-cap)
    evidence_id: str  # REQUIRED provenance — validator rejects if absent/unscreened


class AddClaim(BaseModel):
    """Introduce a new proposition. Starts at the skeptical prior with high u."""

    op: Literal["ADD_CLAIM"] = "ADD_CLAIM"
    text: str
    ontology_tags: list[str]  # must sit within the domain ontology, else FLAG_OOD
    initial_evidence_id: str | None = None


class AddEdge(BaseModel):
    """Assert a typed relation between two existing claims."""

    op: Literal["ADD_EDGE"] = "ADD_EDGE"
    src: str
    dst: str
    type: EdgeType
    weight: float = 1.0


class InvalidateEdge(BaseModel):
    """Retire an edge (bi-temporal: sets invalid_at, never deletes). The reason
    MUST cite an evidence_id — you retract with evidence, not with assertion."""

    op: Literal["INVALIDATE_EDGE"] = "INVALIDATE_EDGE"
    edge_id: str
    evidence_id: str
    reason: str


class FlagOOD(BaseModel):
    """Mark a payload as outside what the ontology represents. Always allowed —
    flagging is safe; it never mutates belief."""

    op: Literal["FLAG_OOD"] = "FLAG_OOD"
    payload: str
    reason: str


class Reject(BaseModel):
    """Refuse a piece of evidence. Always allowed. The correct response to an
    injection payload, malformed data, or an unverifiable claim."""

    op: Literal["REJECT"] = "REJECT"
    evidence_id: str
    reason: RejectReason


# Discriminated union on the "op" tag — Pydantic validates the right shape and
# structured-outputs decoding constrains the model to exactly these.
Op = Annotated[
    ApplyEvidence | AddClaim | AddEdge | InvalidateEdge | FlagOOD | Reject,
    Field(discriminator="op"),
]

OP_NAMES: frozenset[str] = frozenset(
    {"APPLY_EVIDENCE", "ADD_CLAIM", "ADD_EDGE", "INVALIDATE_EDGE", "FLAG_OOD", "REJECT"}
)


class ProposedOps(BaseModel):
    """What the extractor returns for one event. The `think` trace is for the
    audit log / UI only — it is NEVER parsed for actions. Only `ops` reaches the
    validator."""

    think: str = ""  # <think>…</think> rationale, advisory only
    ops: list[Op] = Field(default_factory=list)


def ops_json_schema() -> dict[str, Any]:
    """The JSON Schema for constrained decoding. Feed this to Flash `[train]
    structured_outputs` (GRPO + OPD) and to the serving `response_format` so the
    model literally cannot emit anything outside the vocabulary. One schema,
    train and serve. (Prompt Injection Defense §Layer-3.)"""
    return ProposedOps.model_json_schema()
