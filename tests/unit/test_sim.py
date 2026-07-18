"""Unit tests for the simulator (area B).

These guard the properties the rest of the system relies on: determinism (SFT /
eval reproducibility), PD6 (gold never leaks), gold validity against the closed
op vocabulary, class coverage, and the per-class artifacts that make each gold op
self-evident to a well-behaved extractor + deterministic screen.
"""

from __future__ import annotations

import pytest

from cortesol.core.domain import MIN_PLAUSIBLE_KD_PM, correlation_group
from cortesol.core.ops import ProposedOps
from cortesol.core.schema import EventClass
from cortesol.sim.events import emit_echo_burst, emit_stream
from cortesol.sim.gold import gold_ops
from cortesol.sim.world import World

pytestmark = pytest.mark.unit


def _by_class(seed: int, length: int, cls: EventClass):
    return [e for e in emit_stream(seed, length) if e.sim_meta.event_class is cls]


def test_stream_is_deterministic():
    a = [e.model_dump() for e in emit_stream(7, 21)]
    b = [e.model_dump() for e in emit_stream(7, 21)]
    assert a == b


def test_different_seeds_differ():
    a = [e.raw_text for e in emit_stream(1, 14)]
    b = [e.raw_text for e in emit_stream(2, 14)]
    assert a != b


def test_world_is_deterministic():
    assert [c.id for c in World(5).claims] == [c.id for c in World(5).claims]
    # World always has both true and false binders, and a false efficacy claim
    # (so genuine / hyped classes always have a claim to draw on).
    w = World(0)
    assert w.binders(z=1) and w.binders(z=0)
    assert w.efficacies(z=0)


def test_untrusted_view_strips_gold():
    for e in emit_stream(0, 14):
        assert e.sim_meta is not None  # sim events carry gold
        assert e.untrusted_view().sim_meta is None  # PD6: stripped for the engine


def test_gold_is_valid_closed_vocabulary():
    for e in emit_stream(3, 21):
        po = gold_ops(e)
        assert isinstance(po, ProposedOps)
        assert len(po.ops) >= 1
        # Round-trips through the discriminated op union (no SET_CONFIDENCE/DELETE).
        ProposedOps(ops=e.sim_meta.gold_ops)


def test_all_seven_classes_present_and_balanced():
    classes = [e.sim_meta.event_class for e in emit_stream(0, 14)]
    assert set(classes) == set(EventClass)
    # Two full round-robin cycles -> exactly 2 of each.
    assert all(classes.count(c) == 2 for c in EventClass)


def test_genuine_gold_is_strong_positive():
    for e in _by_class(0, 14, EventClass.GENUINE):
        (op,) = e.sim_meta.gold_ops
        assert op["op"] == "APPLY_EVIDENCE"
        assert op["direction"] == "+" and op["strength"] == "strong"


def test_fraudulent_is_physically_implausible_and_rejected():
    frauds = _by_class(0, 14, EventClass.FRAUDULENT)
    assert frauds
    for e in frauds:
        kd_pm = float(e.fields["value"]) * 1000.0  # nM -> pM
        assert kd_pm < MIN_PLAUSIBLE_KD_PM  # sub-diffusion; the screen must flag it
        assert e.sim_meta.gold_ops[0]["op"] == "REJECT"
        assert e.sim_meta.gold_ops[0]["reason"] == "unverifiable"


def test_injection_payload_only_in_raw_text():
    injs = _by_class(0, 14, EventClass.INJECTION)
    assert injs
    for e in injs:
        assert "SYSTEM" in e.raw_text  # attack sits in the DATA field
        for v in e.fields.values():  # ...and NOWHERE in structured fields
            assert "SYSTEM" not in str(v)
        assert e.sim_meta.gold_ops[0] == {
            "op": "REJECT",
            "evidence_id": e.id,
            "reason": "injection",
        }


def test_out_of_scope_flags_ood_and_moves_no_belief():
    oos = _by_class(0, 14, EventClass.OUT_OF_SCOPE)
    assert oos
    for e in oos:
        (op,) = e.sim_meta.gold_ops
        assert op["op"] == "FLAG_OOD"  # flag, never coerce into a claim
        assert "small-molecule" in op["payload"].lower()
        # OOD events must not carry any belief-moving op or a world-truth claim.
        assert not any(o["op"] == "APPLY_EVIDENCE" for o in e.sim_meta.gold_ops)
        assert e.sim_meta.world_truth == {}


def test_contradictory_is_negative_evidence_on_true_claim():
    for e in _by_class(0, 14, EventClass.CONTRADICTORY):
        (op,) = e.sim_meta.gold_ops
        assert op["op"] == "APPLY_EVIDENCE" and op["direction"] == "-"
        # It targets a genuinely true claim (z=1) — that's what makes it a real
        # contradiction the engine must weigh, not just noise.
        assert e.sim_meta.world_truth.get(op["claim_id"]) == 1


def test_echo_burst_shares_one_correlation_group_then_goes_independent():
    burst = emit_echo_burst(0, k=4)
    assert len(burst) == 5
    groups = [correlation_group(e.fields) for e in burst]
    assert len(set(groups[:-1])) == 1  # the k echoes are one lab|method|dataset
    assert groups[-1] != groups[0]  # the replication is a fresh group
