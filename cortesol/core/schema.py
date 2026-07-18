"""The data model — Claim, Evidence, Source, Edge, RawEvent.

FROZEN CONTRACT (steward: Branch 1). This is the vocabulary of the whole system.
Every branch reads and writes these types; nobody adds a field without a contract
amendment (docs/PROTOCOL.md §Contract Freeze). Terms are defined in
docs/ONTOLOGY.md — read that before touching this file.

Design rules baked into the types:
  * Belief is log-odds `ell`. There is no settable `confidence` field — `c` is a
    read-only view. You cannot assign truth; you can only accumulate evidence.
    (Prime Directive PD2.)
  * Untrusted text lives in `Evidence.raw_text` and data fields ONLY. It never
    occupies an instruction position. (PD3.)
  * Ground-truth / gold labels live in `RawEvent.sim_meta`, which
    `untrusted_view()` strips. The extractor and engine must never see it. (PD6.)
  * Nothing is deleted — claims/edges are retired via status / `invalid_at`. (PD7.)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import (
    PRIOR_C_0,
    SL_BASE_RATE_A,
    SL_PRIOR_WEIGHT_W,
    SOURCE_PRIORS,
)
from .mathx import logit, sigmoid

# --- Enumerations (closed sets — part of the contract) --------------------


class ClaimStatus(str, Enum):  # noqa: UP042 -- frozen wire-contract enum
    ACTIVE = "active"
    FLAGGED = "flagged"  # kept but marked (e.g. OOD, high conflict)
    QUARANTINED = "quarantined"  # writes suspended pending review
    RETIRED = "retired"  # superseded; never deleted


class EdgeType(str, Enum):  # noqa: UP042 -- frozen wire-contract enum
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"  # asymmetric: dst depends on src
    REPLICATES = "replicates"  # strongest positive coupling


Direction = Literal["+", "-"]
Strength = Literal["weak", "moderate", "strong"]


class EventClass(str, Enum):  # noqa: UP042 -- frozen wire-contract enum
    """The simulator's generative classes. The gold op is a pure function of
    (world-state, class). The engine NEVER sees this — it lives in sim_meta and
    is used only to build gold labels and score eval. (Fine-Tuning Plan Stage 0.)
    """

    GENUINE = "genuine"
    NOISY = "noisy"
    HYPED = "hyped"
    FRAUDULENT = "fraudulent"
    CONTRADICTORY = "contradictory"
    OUT_OF_SCOPE = "out_of_scope"
    INJECTION = "injection"


# --- Evidence & Source ----------------------------------------------------


class Source(BaseModel):
    """The origin of a report. Reliability (tau, phi) caps how far any of its
    reports can move belief: |log Lambda_R| <= log(tau / phi). Distrust routes to
    *uncertainty*, not disbelief. (Confidence Math §2.)"""

    id: str
    tier: str = "unknown"  # keys SOURCE_PRIORS
    tau: float = 0.70  # true-report rate  P(R | E)
    phi: float = 0.30  # false-report rate P(R | ~E)
    discredited: bool = False  # retraction / fraud exposed -> TMS cascade
    history: list[str] = Field(default_factory=list)  # evidence ids (Dawid-Skene, stretch)

    @classmethod
    def from_tier(cls, id: str, tier: str) -> Source:
        tau, phi = SOURCE_PRIORS.get(tier, SOURCE_PRIORS["unknown"])
        return cls(id=id, tier=tier, tau=tau, phi=phi)


class Evidence(BaseModel):
    """A single report about the world. DATA, never instructions.

    `raw_text` is quarantined and is never re-executed or placed in an
    instruction position. Structured `fields` are what the engine actually uses.
    `red_flags` come from the deterministic screen and shrink effective Lambda.
    """

    id: str
    source_id: str
    raw_text: str = ""  # quarantined free text
    fields: dict[str, Any] = Field(default_factory=dict)  # effect_size, n, p, venue, lab, method
    red_flags: list[str] = Field(default_factory=list)  # keys RED_FLAG_PHI_BUMP
    correlation_group: str | None = None  # lab x method x dataset -> n_eff damping
    class_estimate: str | None = None  # extractor's guess (advisory only)


# --- Edge (typed, bi-temporal — schema from Graphiti) ---------------------


class Edge(BaseModel):
    """Typed relation between claims. Bi-temporal: retire by setting
    `invalid_at`, never by deleting — enables time-travel queries and honest
    audit history. (System Architecture §data-model.)"""

    id: str
    src: str  # claim id
    dst: str  # claim id
    type: EdgeType
    weight: float = 1.0
    valid_at: int = 0  # event index when asserted
    invalid_at: int | None = None  # event index when invalidated (None = live)

    @property
    def live(self) -> bool:
        return self.invalid_at is None


# --- Claim (the thing we hold belief about) -------------------------------


class TrajectoryPoint(BaseModel):
    t: int  # event index
    ell: float
    cause: str  # human-readable reason for this move (audit)


class Claim(BaseModel):
    """A proposition the KB tracks belief about.

    Belief is `ell` (log-odds). `c`, `u`, and the subjective-logic opinion are
    read-only VIEWS derived from `ell` and the evidence counters (r, s). There is
    deliberately no way to set confidence directly — see PD2 / docs/ONTOLOGY.md.
    """

    id: str
    text: str
    ontology_tags: list[str] = Field(default_factory=list)
    ell: float = Field(default_factory=lambda: logit(PRIOR_C_0))
    r: int = 0  # supporting evidence count
    s: int = 0  # contradicting evidence count
    status: ClaimStatus = ClaimStatus.ACTIVE
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    correlation_seen: dict[str, int] = Field(default_factory=dict)  # group -> count, for n_eff

    # -- read-only views (never persisted as source of truth) --

    @property
    def c(self) -> float:
        """Confidence as a probability. THIS IS NOT TRUTH — it is the system's
        calibrated posterior. (docs/ONTOLOGY.md §Truth vs Belief.)"""
        return sigmoid(self.ell)

    @property
    def u(self) -> float:
        """Subjective-logic uncertainty. A fresh claim has u=1 ('I don't know'),
        structurally different from c=0.5 with lots of conflicting evidence."""
        return SL_PRIOR_WEIGHT_W / (self.r + self.s + SL_PRIOR_WEIGHT_W)

    @property
    def opinion(self) -> dict[str, float]:
        """Jøsang opinion (b, d, u, a) with projected probability P = b + a*u."""
        denom = self.r + self.s + SL_PRIOR_WEIGHT_W
        b = self.r / denom
        d = self.s / denom
        u = SL_PRIOR_WEIGHT_W / denom
        return {"b": b, "d": d, "u": u, "a": SL_BASE_RATE_A, "P": b + SL_BASE_RATE_A * u}


# --- The event stream unit ------------------------------------------------


class SimMeta(BaseModel):
    """Ground-truth sidecar. PRESENT ONLY for simulator / eval streams. Stripped
    by `RawEvent.untrusted_view()`. If any of this reaches the extractor or the
    engine, PD6 is violated and the eval is invalid."""

    event_class: EventClass
    world_truth: dict[str, Any] = Field(default_factory=dict)  # z_j, theta_j snapshot
    gold_ops: list[dict[str, Any]] = Field(default_factory=list)  # the deterministic gold


class RawEvent(BaseModel):
    """One incoming experimental result. The public fields are UNTRUSTED. Only
    `sim_meta` is trusted, and only the eval harness may read it."""

    id: str
    t: int  # position in the stream
    source_id: str
    raw_text: str = ""  # untrusted free text (may contain injection payloads)
    fields: dict[str, Any] = Field(default_factory=dict)  # untrusted structured payload
    sim_meta: SimMeta | None = None  # trusted; None for real CORTEX events

    def untrusted_view(self) -> RawEvent:
        """The ONLY form the extractor / engine may consume. Guarantees no gold
        leakage (PD6). The orchestrator calls this at the quarantine boundary."""
        return RawEvent(
            id=self.id,
            t=self.t,
            source_id=self.source_id,
            raw_text=self.raw_text,
            fields=dict(self.fields),
            sim_meta=None,
        )
