"""Fail-closed inference quorum for checkpointed Cortesol extractors.

Each member sees an independent copy of the same :class:`Context`.  Members may
only propose operations; the pipeline still screens once, validates once, and
commits at most one complete proposal.  In particular, this module never merges
operations from different members (a "Frankenproposal").
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Protocol

from ..core.context import Context
from ..core.ops import AddClaim, ApplyEvidence, InvalidateEdge, ProposedOps, Reject


class Extractor(Protocol):
    """The subset of the extractor interface used by ``process_event``."""

    def extract(self, ctx: Context, model: str | None = None) -> object: ...


Member = Extractor | Callable[[Context], object]

_OP_FIELDS: dict[str, frozenset[str]] = {
    "APPLY_EVIDENCE": frozenset({"op", "claim_id", "direction", "strength", "evidence_id"}),
    "ADD_CLAIM": frozenset({"op", "text", "ontology_tags", "initial_evidence_id"}),
    "ADD_EDGE": frozenset({"op", "src", "dst", "type", "weight"}),
    "INVALIDATE_EDGE": frozenset({"op", "edge_id", "evidence_id", "reason"}),
    "FLAG_OOD": frozenset({"op", "payload", "reason"}),
    "REJECT": frozenset({"op", "evidence_id", "reason"}),
}


def _strict_shape(value: object) -> bool:
    """Reject unknown fields before Pydantic can discard them.

    ``ProposedOps`` is a frozen shared contract whose models intentionally use
    Pydantic's default extra-field behavior.  Network/member payloads need a
    stricter boundary, so the swarm checks the exact wire shape first.
    """
    if not isinstance(value, Mapping) or not set(value).issubset({"think", "ops"}):
        return False
    ops = value.get("ops")
    if not isinstance(ops, list):
        return False
    for op in ops:
        if not isinstance(op, Mapping):
            return False
        op_name = op.get("op")
        allowed = _OP_FIELDS.get(op_name) if isinstance(op_name, str) else None
        if allowed is None or not set(op).issubset(allowed):
            return False
    return True


def _parse_candidate(value: object) -> ProposedOps | None:
    """Parse a member response without accepting loose/coerced wire data."""
    if isinstance(value, ProposedOps):
        proposal = value.model_copy(deep=True)
    else:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not _strict_shape(value):
            return None
        try:
            proposal = ProposedOps.model_validate(value, strict=True)
        except (TypeError, ValueError):
            return None
    # An empty proposal is not a positive vote.  No quorum must fail closed to
    # an auditable REJECT instead of silently doing nothing.
    return proposal if proposal.ops else None


def _eligible_for_event(proposal: ProposedOps, evidence_id: str) -> bool:
    """Read-only preflight for provenance that is knowable from ``Context``.

    Referential integrity, screening, source trust, and all mutation rules stay
    with the authoritative validator after the swarm returns one proposal.
    """
    for op in proposal.ops:
        if isinstance(op, (ApplyEvidence, InvalidateEdge, Reject)):
            if op.evidence_id != evidence_id:
                return False
        elif isinstance(op, AddClaim):
            if op.initial_evidence_id not in {None, evidence_id}:
                return False
    return True


def _canonical(proposal: ProposedOps) -> str:
    """Canonicalize the entire ordered op list, deliberately excluding think."""
    return json.dumps(
        [op.model_dump(mode="json") for op in proposal.ops],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _invoke(member: Member, ctx: Context) -> object:
    isolated = ctx.model_copy(deep=True)
    method = getattr(member, "extract", None)
    if callable(method):
        return method(isolated)
    if callable(member):
        return member(isolated)
    raise TypeError("swarm member must be callable or expose extract(ctx)")


class SwarmExtractor:
    """Run up to three extractor members and require a whole-proposal quorum.

    The quorum is always two votes.  Two-member use is therefore unanimity;
    three-member use tolerates one malformed, timed-out, or dissenting member.
    Completion order cannot affect the selected proposal.
    """

    def __init__(
        self,
        members: Sequence[Member],
        *,
        max_agents: int = 3,
        timeout: float = 180.0,
    ) -> None:
        if not 1 <= max_agents <= 3:
            raise ValueError("max_agents must be between 1 and 3")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not members:
            raise ValueError("at least one swarm member is required")
        self.members = tuple(members[:max_agents])
        self.max_agents = max_agents
        self.timeout = timeout

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        """Return the unique 2-vote proposal, otherwise reject unverifiable."""
        del model  # Members are already bound to explicit checkpoint revisions.
        executor = ThreadPoolExecutor(max_workers=len(self.members), thread_name_prefix="cortesol")
        futures: dict[Future[object], int] = {
            executor.submit(_invoke, member, ctx): index
            for index, member in enumerate(self.members)
        }
        done, pending = wait(futures, timeout=self.timeout)
        for future in pending:
            future.cancel()
        # Do not let a slow member extend the swarm's wall-clock timeout.  Every
        # member received an isolated Context, so a late completion cannot alter
        # the caller's state or the vote already being computed.
        executor.shutdown(wait=False, cancel_futures=True)

        candidates: list[tuple[str, ProposedOps]] = []
        # Stable member-index order makes exception/result processing independent
        # of thread completion order.
        for future in sorted(done, key=futures.__getitem__):
            try:
                proposal = _parse_candidate(future.result())
            except Exception:
                proposal = None
            if proposal is None or not _eligible_for_event(proposal, ctx.evidence.id):
                continue
            candidates.append((_canonical(proposal), proposal))

        counts = Counter(key for key, _ in candidates)
        winners = sorted(key for key, count in counts.items() if count >= 2)
        if len(winners) == 1:
            # Re-parse the canonical wire form to eliminate rationale and ensure
            # the returned object is independent of every member-owned object.
            ops = json.loads(winners[0])
            return ProposedOps.model_validate({"ops": ops}, strict=True)
        return ProposedOps(
            ops=[Reject(evidence_id=ctx.evidence.id, reason="unverifiable")]
        )
