"""Multi-domain / field-independence tests (area B).

Two guarantees the paid multi-domain training run depends on:
  1. The PEPTIDE path is byte-identical to the pre-spec simulator (frozen hashes +
     the existing suite), so no peptide dataset or downstream artifact shifts.
  2. Two toy non-peptide fields (materials, ml_benchmarks) exercise all seven event
     classes, produce valid closed-vocabulary gold, and trip the deterministic
     screen the same way — so a Freesolo model can learn a field-INDEPENDENT policy.
"""

from __future__ import annotations

import hashlib
import json
import tempfile

import pytest

from cortesol.core.domains import get_active_domain, set_active_domain
from cortesol.core.kb import KB
from cortesol.core.ops import OP_NAMES, ProposedOps
from cortesol.core.schema import Claim, EventClass
from cortesol.ingest.quarantine import quarantine
from cortesol.ingest.screen import screen
from cortesol.sim.events import emit_stream
from cortesol.sim.gold import gold_ops
from cortesol.sim.specs import SPECS
from cortesol.sim.world import World
from cortesol.train.datasets import build_all, build_multidomain, build_multidomain_sft_rows

pytestmark = pytest.mark.unit

TOY_DOMAINS = ("materials", "ml_benchmarks")

# Frozen peptide reference hashes — a guard that the spec refactor stays
# byte-identical for the default field. (Regenerate ONLY on a deliberate change.)
_PEPTIDE_STREAM_SHA = {
    (0, 14): "7bae48d53e870e8deca7a94e28c533ff70e671d51846a3d1056c8a361657f946",
    (7, 21): "6e9abbe8dc1f1724f00793740aaa785dd0ac12f5f82ac9efe1acb3234ac5ba44",
    (202, 28): "f81b2fbf3782f70c9dd2d02dee41fed11c6ff1a79d2568054fdb7a3fd943b564",
}


@pytest.fixture(autouse=True)
def _restore_active_domain():
    """The active domain is process-global; never leak a switch into other tests."""
    original = get_active_domain()
    yield
    set_active_domain(original)


def _stream_sha(seed: int, length: int) -> str:
    blob = "\n".join(e.model_dump_json() for e in emit_stream(seed, length))
    return hashlib.sha256(blob.encode()).hexdigest()


@pytest.mark.parametrize(("seed", "length"), list(_PEPTIDE_STREAM_SHA))
def test_peptide_stream_is_byte_identical_to_frozen_reference(seed, length):
    assert _stream_sha(seed, length) == _PEPTIDE_STREAM_SHA[(seed, length)]


def _seed_kb(spec) -> KB:
    kb = KB()
    for wc in World(0, entity_prefix=spec.entity_prefix, spec=spec).claims:
        kb.add_claim(Claim(id=wc.id, text=wc.text, ontology_tags=list(wc.ontology_tags)))
    return kb


@pytest.mark.parametrize("name", TOY_DOMAINS)
def test_toy_domain_exercises_all_seven_classes_with_valid_gold(name):
    spec = SPECS[name]
    events = emit_stream(0, 14, entity_prefix=spec.entity_prefix, spec=spec)
    assert {e.sim_meta.event_class for e in events} == set(EventClass)
    for e in events:
        po = gold_ops(e)
        assert isinstance(po, ProposedOps) and len(po.ops) >= 1
        ProposedOps(ops=e.sim_meta.gold_ops)  # round-trips the closed op union


@pytest.mark.parametrize("name", TOY_DOMAINS)
def test_toy_domain_is_deterministic_and_differs_from_peptides(name):
    spec = SPECS[name]
    a = [e.model_dump() for e in emit_stream(3, 14, entity_prefix=spec.entity_prefix, spec=spec)]
    b = [e.model_dump() for e in emit_stream(3, 14, entity_prefix=spec.entity_prefix, spec=spec)]
    assert a == b
    peptide = [e.model_dump() for e in emit_stream(3, 14)]
    assert a != peptide  # a genuinely different field of knowledge


@pytest.mark.parametrize("name", TOY_DOMAINS)
def test_toy_domain_trips_the_screen_the_same_way(name):
    spec = SPECS[name]
    set_active_domain(spec.core_domain)
    kb = _seed_kb(spec)
    required = {
        EventClass.FRAUDULENT: "value_physically_implausible",
        EventClass.INJECTION: "prompt_injection",
        EventClass.OUT_OF_SCOPE: "out_of_scope",
    }
    safe = {EventClass.GENUINE, EventClass.NOISY, EventClass.CONTRADICTORY, EventClass.INJECTION}
    for e in emit_stream(0, 14, entity_prefix=spec.entity_prefix, spec=spec):
        flags = screen(quarantine(e), kb)
        cls = e.sim_meta.event_class
        if cls in required:
            assert required[cls] in flags, f"{name}/{cls.value}: missing {required[cls]} in {flags}"
        # a plausible primary-metric report must never look physically impossible
        if cls in safe:
            assert "value_physically_implausible" not in flags


@pytest.mark.parametrize("name", TOY_DOMAINS)
def test_toy_domain_injection_payload_only_in_raw_text(name):
    spec = SPECS[name]
    injs = [
        e
        for e in emit_stream(0, 14, entity_prefix=spec.entity_prefix, spec=spec)
        if e.sim_meta.event_class is EventClass.INJECTION
    ]
    assert injs
    for e in injs:
        assert "SYSTEM" in e.raw_text
        for v in e.fields.values():
            assert "SYSTEM" not in str(v)
        assert e.sim_meta.gold_ops[0]["op"] == "REJECT"
        assert e.sim_meta.gold_ops[0]["reason"] == "injection"


@pytest.mark.parametrize("name", TOY_DOMAINS)
def test_toy_domain_out_of_scope_reason_comes_from_domain(name):
    spec = SPECS[name]
    oos = [
        e
        for e in emit_stream(0, 14, entity_prefix=spec.entity_prefix, spec=spec)
        if e.sim_meta.event_class is EventClass.OUT_OF_SCOPE
    ]
    assert oos
    for e in oos:
        (op,) = e.sim_meta.gold_ops
        assert op["op"] == "FLAG_OOD"
        assert op["reason"] == spec.oos_reason
        assert "peptide" not in op["reason"]  # not peptide-flavored
        assert e.sim_meta.world_truth == {}


def test_multidomain_build_covers_every_domain_class_and_op(tmp_path):
    manifest = build_multidomain(tmp_path, seeds_per_domain=3, events_per_seed=14)
    rows = [
        json.loads(line)
        for line in (tmp_path / "sft_train_multidomain.jsonl").read_text().splitlines()
    ]
    # transport shape + no gold leakage
    for row in rows:
        assert set(row) == {"input", "output", "metadata"}
        assert "sim_meta" not in row["input"]

    # every (domain, class) is present and balanced
    counts = manifest["rows_per_domain_class"]
    assert set(counts) == set(manifest["domains"])
    for domain, per_class in counts.items():
        assert {c for c in per_class} == {c.value for c in EventClass}, domain

    # full op coverage across the mixed dataset (structural suites contribute the
    # structural ops per field)
    ops = {op["op"] for row in rows for op in json.loads(row["output"])["ops"]}
    assert ops == set(OP_NAMES)

    # domains are tagged and disjoint by case_id
    assert {row["metadata"]["domain"] for row in rows} == set(manifest["domains"])
    assert len({row["metadata"]["case_id"] for row in rows}) == len(rows)


def test_multidomain_build_is_deterministic_and_restores_active_domain(tmp_path):
    before = get_active_domain().name
    first = build_multidomain(tmp_path / "a", seeds_per_domain=2, events_per_seed=14)
    assert get_active_domain().name == before  # restored despite per-field switching
    second = build_multidomain(tmp_path / "b", seeds_per_domain=2, events_per_seed=14)
    assert (
        first["files"]["sft_train_multidomain"]["sha256"]
        == second["files"]["sft_train_multidomain"]["sha256"]
    )


def test_multidomain_rejects_unknown_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        build_multidomain_sft_rows(["peptides", "astrology"], seeds_per_domain=1, events_per_seed=7)


def test_multidomain_build_does_not_change_the_default_peptide_dataset(tmp_path):
    """Building the multi-domain file leaves the frozen peptide artifacts untouched."""
    with tempfile.TemporaryDirectory() as d:
        baseline = {n: it["sha256"] for n, it in build_all(d)["files"].items()}
    build_multidomain(tmp_path, seeds_per_domain=2, events_per_seed=7)
    with tempfile.TemporaryDirectory() as d:
        after = {n: it["sha256"] for n, it in build_all(d)["files"].items()}
    assert baseline == after
