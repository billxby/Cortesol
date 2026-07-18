"""Offline tests for the real Flash extractor seam (`ingest/extract.py::extract`).

No network: the OpenAI client is monkeypatched with a fake that returns canned
content. Asserts (1) a valid reply parses into ProposedOps, (2) the op schema is
passed as the constrained-decoding response_format, (3) a malformed reply or a
missing model is FAIL-SAFE (zero ops, never raises)."""

from __future__ import annotations

import types

from cortesol.core.context import Context
from cortesol.core.ops import ApplyEvidence, ProposedOps, ops_json_schema
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


def test_valid_reply_parses_and_passes_the_op_schema(monkeypatch):
    captured: dict = {}
    good = ProposedOps(
        think="clean binding result",
        ops=[ApplyEvidence(claim_id="c_bind_P1_MC4R", direction="+", strength="strong", evidence_id="e0")],
    ).model_dump_json()
    monkeypatch.setattr(ex, "_get_client", lambda: _fake_client(good, captured))

    out = ex.extract(_ctx(), model="test-model")

    assert isinstance(out, ProposedOps)
    assert len(out.ops) == 1 and out.ops[0].op == "APPLY_EVIDENCE"
    # constrained decoding: the exact op schema is the response_format
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == ops_json_schema()
    assert captured["model"] == "test-model"


def test_malformed_reply_is_fail_safe(monkeypatch):
    monkeypatch.setattr(ex, "_get_client", lambda: _fake_client("not json at all", {}))
    out = ex.extract(_ctx(), model="test-model")
    assert isinstance(out, ProposedOps)
    assert out.ops == []  # no belief-moving ops on a bad reply


def test_empty_reply_is_fail_safe(monkeypatch):
    monkeypatch.setattr(ex, "_get_client", lambda: _fake_client("", {}))
    out = ex.extract(_ctx(), model="test-model")
    assert out.ops == []


def test_no_model_configured_returns_no_ops(monkeypatch):
    monkeypatch.setattr(ex, "_resolve_model", lambda m: None)
    out = ex.extract(_ctx())
    assert out.ops == []
