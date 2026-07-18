"""Tests for deterministic field extraction from abstract prose + the screen /
quarantine wiring that consumes it. Verifies the simulator path is untouched."""

from __future__ import annotations

from cortesol.ingest.fieldparse import parse_fields
from cortesol.ingest.quarantine import quarantine
from cortesol.ingest.screen import screen
from cortesol.core.kb import KB
from cortesol.core.schema import RawEvent, Source


def test_parses_rct_stats_and_design():
    text = (
        "A randomized, double-blind, placebo-controlled trial of semaglutide in "
        "1,961 adults with obesity. Mean weight loss was 14.9% (p<0.001)."
    )
    f = parse_fields(text)
    assert f["study_type"] == "clinical"
    assert f["randomized"] is True and f["blinded"] is True and f["controlled"] is True
    assert f["n"] == 1961
    assert f["p"] < 0.001 + 1e-9
    assert f["metric"] == "percent_change" and f["units"] == "percent"


def test_parses_meta_analysis_as_review():
    f = parse_fields("A systematic review and meta-analysis of 4 randomized controlled trials.")
    assert f["study_type"] == "review"


def test_parses_case_report_and_affinity_units():
    f = parse_fields("We report a case of angioedema. Separately, Kd = 5 µM was measured.")
    assert f.get("case_report") is True
    # µM normalised to nM
    assert f["metric"] == "Kd" and f["units"] == "nM" and f["value"] == 5000.0


def test_bare_letters_do_not_false_match_affinity():
    # 'Ki' inside 'taking' / 'Kd' nowhere — no metric without a number+unit
    f = parse_fields("Patients taking the drug showed improvement.")
    assert "metric" not in f or f["metric"] != "Ki"


def test_quarantine_enriches_only_when_no_structured_metric():
    # real paper: no fields.metric -> parsed from raw_text
    paper = RawEvent(
        id="pmid_1", t=0, source_id="v",
        raw_text="Randomized placebo-controlled trial; n=800; weight loss 12% (p<0.001).",
        fields={"peptide": "semaglutide", "primary_target": "GLP1R"},
    )
    ev = quarantine(paper)
    assert ev.fields["study_type"] == "clinical"
    assert ev.fields["n"] == 800

    # simulator event already HAS a metric -> parser must not run / not override
    sim = RawEvent(
        id="e0", t=0, source_id="lab", raw_text="SPR: P1 binds MC4R, Kd = 27.9 nM (n=3).",
        fields={"metric": "Kd", "value": 27.9, "units": "nM", "n": 3},
    )
    ev2 = quarantine(sim)
    assert ev2.fields == {"metric": "Kd", "value": 27.9, "units": "nM", "n": 3}


def test_screen_flags_uncontrolled_clinical_but_not_strong_rct():
    kb = KB()
    kb.add_source(Source.from_tier("v", "reputable"))

    weak = quarantine(RawEvent(
        id="p1", t=0, source_id="v",
        raw_text="Open-label observational study of 6 patients; weight loss 9%.",
        fields={"peptide": "x", "primary_target": "y"},
    ))
    weak_flags = screen(weak, kb)
    assert "uncontrolled" in weak_flags and "underpowered" in weak_flags
    assert "purity_not_reported" not in weak_flags  # not a binding assay

    strong = quarantine(RawEvent(
        id="p2", t=0, source_id="v",
        raw_text="Randomized double-blind placebo-controlled trial; n=1,961; 15% loss (p<0.001).",
        fields={"peptide": "x", "primary_target": "y"},
    ))
    strong_flags = screen(strong, kb)
    assert "uncontrolled" not in strong_flags and "underpowered" not in strong_flags
