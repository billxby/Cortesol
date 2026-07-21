"""Unit tests for the appraisal SFT dataset factory (`train/appraisal_dataset`).

Deterministic and NETWORK-FREE (StubTeacher only). Coverage: the corpus merge
(field-tagged), the SFT builder (rows target the appraisal grammar; field carries
no intake signal), balanced adversarial augmentation, held-out-FIELD disjointness,
and the curated known-cases proxy ground truth."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from cortesol.core.assessment import EvidenceAssessment, assessment_ordered_regex
from cortesol.core.domains import get_active_domain, set_active_domain
from cortesol.train import teacher_appraise as ta
from cortesol.train.appraisal_dataset import (
    HELDOUT_FIELDS,
    AppraisalExample,
    CorpusPaper,
    build_adversarial,
    build_appraisal_all,
    build_appraisal_sft,
    build_appraisal_sft_rows,
    build_corpus,
    load_known_cases,
    sft_row,
    split_examples_by_field,
)

pytestmark = pytest.mark.unit

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_ORDERED = re.compile(assessment_ordered_regex())
_STRONG_POSITIVE = {("APPLY_EVIDENCE", "+", "strong"), ("APPLY_EVIDENCE", "+", "moderate")}
_FIELDS = ("physics", "cs", "math", "econ", "stat", "q-bio", "biomed")


@pytest.fixture(autouse=True)
def _peptides_active():
    original = get_active_domain()
    set_active_domain("peptides")
    yield
    set_active_domain(original)


def _corpus() -> list[CorpusPaper]:
    papers: list[CorpusPaper] = []
    for field in _FIELDS:
        for j in range(4):  # >= len(TRANSFORMS) so every transform has a distinct paper
            papers.append(
                CorpusPaper(
                    paper_id=f"{field}:{j}",
                    field=field,
                    title=f"{field} study {j}",
                    abstract=f"A {field} study measured an effect (n={100 + j}, p=0.002).",
                    source_tier="preprint",
                )
            )
    return papers


def test_build_corpus_merges_fields_from_arxiv_and_pubmed():
    arxiv = [{"arxiv_id": "2401.1", "field": "physics", "title": "T", "abstract": "Abs one."}]
    pubmed = [{"pmid": "123", "title": "T", "abstract": "Abs two.", "journal": "Nature"}]
    corpus = build_corpus(arxiv, pubmed)
    fields = {p.field for p in corpus}
    assert fields == {"physics", "biomed"}
    by_id = {p.paper_id: p for p in corpus}
    assert by_id["arxiv:2401.1"].source_tier == "preprint"
    assert by_id["pmid:123"].field == "biomed"
    assert by_id["pmid:123"].source_tier == "top_journal"  # Nature -> prestige tier


def test_sft_rows_target_the_appraisal_grammar():
    examples, _ = build_appraisal_sft(_corpus(), appraiser=ta.StubTeacher())
    rows = build_appraisal_sft_rows(examples)
    assert rows
    for row in rows:
        assert set(row) == {"input", "output", "metadata"}
        assert "sim_meta" not in row["input"]
        # output is the canonical assessment and matches the ordered decoding grammar
        assert _ORDERED.fullmatch(row["output"])
        EvidenceAssessment.model_validate_json(row["output"], strict=True)
        assert row["metadata"]["field"] in _FIELDS
        assert row["metadata"]["schema_version"] == "2.0"


def test_intake_carries_no_field_signal():
    # identical (evidence_id, text, tier) but different field -> identical intake, so
    # the field (subject/discipline) cannot be a label shortcut.
    asmt = ta.stub_appraise("Same abstract text with n=100 and p=0.01.", "same")
    e1 = AppraisalExample("same", "physics", "Same abstract text with n=100 and p=0.01.",
                          "preprint", asmt, "real")
    e2 = AppraisalExample("same", "math", "Same abstract text with n=100 and p=0.01.",
                          "preprint", asmt, "real")
    r1, r2 = sft_row(e1), sft_row(e2)
    assert r1["input"] == r2["input"]
    assert r1["metadata"]["field"] != r2["metadata"]["field"]
    assert "corpus_preprint" in r1["input"]  # neutral, tier-based source id


def test_adversarial_augmentation_is_balanced_across_fields():
    corpus = _corpus()
    adv = build_adversarial(corpus, per_field_per_transform=1)
    kinds_by_field: dict[str, set[str]] = defaultdict(set)
    for example in adv:
        kinds_by_field[example.field].add(example.kind)
    expected = {
        "adversarial:injection_attack",
        "adversarial:overclaiming",
        "adversarial:case_report",
        "adversarial:extraordinary",
    }
    for field in _FIELDS:
        assert kinds_by_field[field] == expected, field


def test_held_out_field_split_is_disjoint():
    examples, _ = build_appraisal_sft(_corpus(), appraiser=ta.StubTeacher())
    train, test = split_examples_by_field(examples)
    train_fields = {e.field for e in train}
    test_fields = {e.field for e in test}
    assert train_fields.isdisjoint(test_fields)
    assert test_fields == set(HELDOUT_FIELDS)
    assert not (set(HELDOUT_FIELDS) & train_fields)


def test_build_appraisal_all_writes_deterministic_splits(tmp_path):
    corpus = _corpus()
    first = build_appraisal_all(tmp_path / "a", corpus=corpus, appraiser=ta.StubTeacher())
    second = build_appraisal_all(tmp_path / "b", corpus=corpus, appraiser=ta.StubTeacher())
    assert first["files"]["appraisal_sft_train"]["sha256"] == (
        second["files"]["appraisal_sft_train"]["sha256"]
    )
    assert first["heldout_fields"] == list(HELDOUT_FIELDS)
    assert set(first["train_fields"]).isdisjoint(HELDOUT_FIELDS)
    # the held-out split only contains held-out fields
    heldout_rows = [
        json.loads(line)
        for line in (tmp_path / "a" / "appraisal_sft_heldout.jsonl").read_text().splitlines()
    ]
    assert {r["metadata"]["field"] for r in heldout_rows} <= set(HELDOUT_FIELDS)


def test_known_cases_are_proxy_ground_truth():
    cases = load_known_cases(_DATA_DIR)
    assert cases, "known_cases.jsonl should ship with the repo"
    for case in cases:
        appraisal = ta.stub_appraise(case["text"], case["case_id"])
        signature = ta.op_signature(appraisal, case["text"])
        if case["expected_belief"] == "high":
            assert signature in _STRONG_POSITIVE, (case["case_id"], signature)
        else:
            assert signature not in _STRONG_POSITIVE, (case["case_id"], signature)
