"""Contract tests — the invariants that must hold on every branch, always.

These exercise the FILLED-IN shared contracts (schema, ops, domain, config,
mathx) and the fixture. They are the merge gate: `make test-contract`. As the
engine/validator/etc. get implemented, add their invariants here too (bounded
step, no-op-without-provenance, gold-never-leaks...).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cortesol.core import config, domain
from cortesol.core.mathx import kish_neff, logit, neff_marginal_factor, sigmoid
from cortesol.core.ops import ProposedOps, ops_json_schema
from cortesol.core.schema import Claim, RawEvent

pytestmark = pytest.mark.contract

FIXTURE = Path(__file__).parents[1] / "fixtures" / "toy_peptide_stream.jsonl"


def test_fresh_claim_is_maximally_uncertain_and_skeptical():
    """A brand-new claim says 'I don't know' (u=1), not '50/50', and starts at the
    skeptical prior — not at whatever an abstract asserts."""
    c = Claim(id="c1", text="P7 binds GLP1R")
    assert c.u == 1.0
    assert c.r == 0 and c.s == 0
    assert c.c == pytest.approx(config.PRIOR_C_0, abs=1e-6)


def test_log_odds_roundtrip():
    for p in (0.1, 0.45, 0.7, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)


def test_neff_discounts_echoes():
    """Independent evidence: full weight. Correlated echoes: shrinking marginal
    weight. This is the double-counting defense."""
    assert kish_neff(1, config.RHO_WITHIN_GROUP) == pytest.approx(1.0)
    assert neff_marginal_factor(0, config.RHO_WITHIN_GROUP) == 1.0
    assert neff_marginal_factor(3, config.RHO_WITHIN_GROUP) < neff_marginal_factor(
        1, config.RHO_WITHIN_GROUP
    )


def test_source_cap_orders_by_reliability():
    """The anti-hype guarantee: a weak source's cap is far below a top source's,
    so it cannot produce a big belief jump no matter how sensational its content."""

    def cap(tier: str) -> float:
        tau, phi = config.SOURCE_PRIORS[tier]
        return math.log(tau / phi)

    assert cap("weak") < cap("preprint") < cap("top_journal")
    assert cap("weak") < config.DELTA_MAX  # a lone weak report can't max out the bound


def test_op_vocabulary_is_closed_and_has_a_schema():
    """No SET_CONFIDENCE, no DELETE — the deliberate absences are the security
    boundary. And the schema exists for constrained decoding at train+serve."""
    assert "SET_CONFIDENCE" not in domain_ops_names()
    assert "DELETE" not in domain_ops_names()
    schema = ops_json_schema()
    assert "ops" in schema.get("properties", {})


def domain_ops_names() -> set[str]:
    from cortesol.core.ops import OP_NAMES

    return set(OP_NAMES)


def test_untrusted_view_strips_ground_truth():
    """PD6: the extractor/engine may only ever see untrusted_view(); sim_meta
    (world truth + gold ops) must not survive it."""
    ev = RawEvent(
        id="e",
        t=0,
        source_id="s",
        sim_meta={"event_class": "genuine", "world_truth": {"x": 1}, "gold_ops": []},
    )
    assert ev.sim_meta is not None
    assert ev.untrusted_view().sim_meta is None


def test_fixture_parses_and_gold_is_valid_ops():
    """Every fixture event parses as a RawEvent, and every gold op it carries is a
    real op in the closed vocabulary."""
    lines = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    assert len(lines) == 6
    classes = set()
    for row in lines:
        ev = RawEvent(**row)
        assert ev.sim_meta is not None
        classes.add(ev.sim_meta.event_class.value)
        # gold ops must validate against the op vocabulary
        ProposedOps(ops=ev.sim_meta.gold_ops)
    assert {
        "genuine",
        "hyped",
        "fraudulent",
        "injection",
        "out_of_scope",
        "contradictory",
    } <= classes


def test_correlation_group_is_lab_method_dataset():
    g = domain.correlation_group({"lab": "lab_A", "method": "SPR", "dataset": "ds1"})
    assert g == "lab_A|SPR|ds1"
