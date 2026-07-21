"""Offline tests for the real Flash extractor seam (`ingest/extract.py::extract`).

No network: the OpenAI client is monkeypatched with a fake that returns canned
content. Asserts (1) a valid reply parses into an EvidenceAssessment, (2) the
assessment schema is passed as the constrained-decoding response_format, (3) a
malformed reply or a missing model is FAIL-SAFE — it returns `_empty_assessment`
(a valid neutral form the judge compiles to a safe FLAG_OOD), never raising."""

from __future__ import annotations

import types

from cortesol.core.assessment import EvidenceAssessment, assessment_json_schema
from cortesol.core.context import Context
from cortesol.core.schema import Claim, Evidence, RawEvent
from cortesol.ingest import extract as ex


def _ctx() -> Context:
    event = RawEvent(id="e0", t=0, source_id="s", raw_text="P1 binds MC4R")
    evidence = Evidence(id="e0", source_id="s", raw_text="P1 binds MC4R", fields={})
    return Context(
        event=event,
        evidence=evidence,
        claims=[Claim(id="c_bind_P1_MC4R", text="P1 binds MC4R")],
        sources=[],
        edges=[],
    )


def _valid_assessment_json() -> str:
    return EvidenceAssessment(
        schema_version="2.0",
        evidence_id="e0",
        in_scope="in_scope",
        instruction_attack=False,
        document_type="primary_study",
        study_design="randomized_controlled",
        subject="P1",
        object="MC4R",
        claim_summary="P1 binds MC4R",
        claim_direction="supports",
        magnitude="Kd 12 nM",
        value=12.0,
        value_relation="approximate",
        units="nM",
        sample_size=3,
        replicate_count=3,
        p_value=0.001,
        confidence_interval_reported=True,
        effect_size_reported=True,
        controlled=True,
        randomized=True,
        blinded=True,
        preregistered=True,
        independent_replication=True,
        extraordinary_claim=False,
        overclaiming=False,
    ).model_dump_json()


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, content: str, captured: dict) -> None:
        self._content, self._captured = content, captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeResp(self._content)


def _fake_client(content: str, captured: dict):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions(content, captured))
    )


def test_valid_reply_parses_and_passes_the_assessment_schema(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(ex, "_get_client", lambda: _fake_client(_valid_assessment_json(), captured))

    out = ex.extract(_ctx(), model="test-model")

    assert isinstance(out, EvidenceAssessment)
    assert out.evidence_id == "e0"
    assert out.in_scope == "in_scope"
    assert out.study_design == "randomized_controlled"
    # constrained decoding: the exact assessment schema is the response_format
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "EvidenceAssessment"
    assert rf["json_schema"]["schema"] == assessment_json_schema()
    assert captured["model"] == "test-model"


def test_malformed_reply_is_fail_safe(monkeypatch):
    monkeypatch.setattr(ex, "_get_client", lambda: _fake_client("not json at all", {}))
    out = ex.extract(_ctx(), model="test-model")
    # a bad reply falls back to the neutral empty form (never raises, never a claim edit)
    assert isinstance(out, EvidenceAssessment)
    assert out == ex._empty_assessment(_ctx())
    assert out.in_scope == "unclear"
    assert out.claim_direction == "not_a_claim"


def test_empty_reply_is_fail_safe(monkeypatch):
    monkeypatch.setattr(ex, "_get_client", lambda: _fake_client("", {}))
    out = ex.extract(_ctx(), model="test-model")
    assert isinstance(out, EvidenceAssessment)
    assert out == ex._empty_assessment(_ctx())


def test_no_model_configured_returns_empty_assessment(monkeypatch):
    monkeypatch.setattr(ex, "_resolve_model", lambda m: None)
    out = ex.extract(_ctx())
    assert isinstance(out, EvidenceAssessment)
    assert out == ex._empty_assessment(_ctx())
