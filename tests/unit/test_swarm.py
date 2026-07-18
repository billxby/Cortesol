from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from cortesol.core.context import Context
from cortesol.core.ops import ProposedOps, Reject
from cortesol.core.schema import Evidence, RawEvent
from cortesol.ingest.swarm import SwarmExtractor


def _context() -> Context:
    event = RawEvent(id="world:event-1", t=1, source_id="world:source-1", raw_text="P7 binds")
    evidence = Evidence(
        id=event.id,
        event_id=event.id,
        source_id=event.source_id,
        raw_text=event.raw_text,
    )
    return Context(event=event, evidence=evidence)


def _proposal(*, think: str = "") -> ProposedOps:
    return ProposedOps(
        think=think,
        ops=[Reject(evidence_id="world:event-1", reason="unverifiable")],
    )


def _delayed(delay: float, value: object) -> Callable[[Context], object]:
    def member(ctx: Context) -> object:
        time.sleep(delay)
        return value

    return member


@pytest.mark.unit
@pytest.mark.parametrize(
    "members",
    [
        [_delayed(0.03, _proposal(think="a")), _delayed(0.01, _proposal(think="b"))],
        [_delayed(0.01, _proposal(think="a")), _delayed(0.03, _proposal(think="b"))],
    ],
)
def test_quorum_ignores_think_and_arrival_order(members: list[Callable[[Context], object]]) -> None:
    result = SwarmExtractor(members, timeout=1).extract(_context())

    assert result == ProposedOps(ops=[Reject(evidence_id="world:event-1", reason="unverifiable")])
    assert result.think == ""


@pytest.mark.unit
def test_malformed_malicious_and_wrong_provenance_fail_closed() -> None:
    members: list[Callable[[Context], object]] = [
        lambda ctx: "not json",
        lambda ctx: {
            "ops": [{"op": "SET_CONFIDENCE", "claim_id": "c", "value": 1.0}]
        },
        lambda ctx: {
            "ops": [
                {"op": "REJECT", "evidence_id": "forged:event", "reason": "unverifiable"}
            ]
        },
    ]

    result = SwarmExtractor(members).extract(_context())

    assert result == ProposedOps(ops=[Reject(evidence_id="world:event-1", reason="unverifiable")])


@pytest.mark.unit
def test_no_quorum_never_builds_a_frankenproposal() -> None:
    apply = {
        "op": "APPLY_EVIDENCE",
        "claim_id": "c_bind_P7_GLP1R",
        "direction": "+",
        "strength": "moderate",
        "evidence_id": "world:event-1",
    }
    flag = {"op": "FLAG_OOD", "payload": "P7", "reason": "unmapped"}
    members: list[Callable[[Context], object]] = [
        lambda ctx: {"ops": [apply]},
        lambda ctx: {"ops": [flag]},
        lambda ctx: {"ops": [apply, flag]},
    ]

    result = SwarmExtractor(members).extract(_context())

    assert result == ProposedOps(ops=[Reject(evidence_id="world:event-1", reason="unverifiable")])


@pytest.mark.unit
def test_member_contexts_are_isolated_and_timeout_abstains() -> None:
    original = _context()

    def mutating_member(ctx: Context) -> ProposedOps:
        ctx.evidence.id = "tampered"
        return _proposal()

    result = SwarmExtractor(
        [mutating_member, _delayed(0.2, _proposal()), lambda ctx: _proposal()],
        timeout=0.05,
    ).extract(original)

    assert original.evidence.id == "world:event-1"
    assert result == ProposedOps(ops=[Reject(evidence_id="world:event-1", reason="unverifiable")])


@pytest.mark.unit
def test_strict_wire_shape_rejects_hidden_extra_fields() -> None:
    suspicious = {
        "ops": [
            {
                "op": "REJECT",
                "evidence_id": "world:event-1",
                "reason": "unverifiable",
                "set_confidence": 1.0,
            }
        ]
    }
    result = SwarmExtractor([lambda ctx: suspicious, lambda ctx: suspicious]).extract(_context())

    assert result == ProposedOps(ops=[Reject(evidence_id="world:event-1", reason="unverifiable")])


@pytest.mark.unit
def test_configuration_caps_members_at_three() -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        SwarmExtractor([lambda ctx: _proposal()], max_agents=4)
