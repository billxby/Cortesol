"""Offline tests for real-paper ingestion + ADD_CLAIM materialization.

Covers `adapters/cortex.py` (load_stream / journal_to_tier / seed_kb_from_papers)
and the engine's ADD_CLAIM handling (a new node is created at the skeptical prior,
belief untouched)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortesol.adapters.cortex import journal_to_tier, load_stream, seed_kb_from_papers
from cortesol.core import config
from cortesol.core.engine import apply, claim_id_for
from cortesol.core.kb import KB
from cortesol.core.mathx import logit
from cortesol.core.ops import AddClaim
from cortesol.core.schema import RawEvent

_PAPERS = Path(__file__).resolve().parents[2] / "data" / "papers" / "raw_papers.jsonl"


def test_load_stream_parses_real_papers():
    events = load_stream(_PAPERS)
    assert len(events) > 100
    assert all(isinstance(e, RawEvent) for e in events)
    e0 = events[0]
    assert e0.id.startswith("pmid_")
    assert e0.raw_text  # title + abstract
    assert e0.sim_meta is None  # real events carry no gold
    assert e0.t == 0 and events[1].t == 1  # sequential


def test_journal_to_tier_orders_by_reputation():
    assert journal_to_tier("The New England journal of medicine") == "top_journal"
    assert journal_to_tier("bioRxiv") == "preprint"
    assert journal_to_tier("Circulation") == "reputable"
    # An unrecognised but PUBLISHED venue defaults to `reputable`, never `unknown`:
    # a peer-reviewed paper is not epistemically equivalent to knowing nothing.
    # Distrust is still expressed by the lower preprint/predatory caps.
    assert journal_to_tier("Some Unheard-of Venue") == "reputable"
    # every tier returned must be a real SOURCE_PRIORS key
    for j in ("Nature", "medRxiv", "Drugs", "xyz"):
        assert journal_to_tier(j) in config.SOURCE_PRIORS


def test_seed_kb_from_papers_builds_claims_sources_edges():
    events = load_stream(_PAPERS)
    kb = seed_kb_from_papers(events)
    assert len(kb.claims) > 0
    assert len(kb.sources) > 0
    # every seeded binding claim starts at the skeptical prior (no belief set)
    prior = logit(config.PRIOR_C_0)
    assert all(abs(c.ell - prior) < 1e-9 for c in kb.claims.values())


def test_add_claim_materialises_at_prior():
    kb = KB()
    op = AddClaim(text="semaglutide is efficacious in obesity",
                  ontology_tags=["peptide:semaglutide", "efficacy:obesity"])
    apply(kb, op, None)
    cid = claim_id_for(op.text)
    assert cid in kb.claims
    c = kb.claims[cid]
    assert c.c == pytest.approx(config.PRIOR_C_0, abs=1e-6)  # prior, high u
    assert c.u == 1.0
    # idempotent — re-adding the same text does not duplicate
    apply(kb, op, None)
    assert len(kb.claims) == 1
