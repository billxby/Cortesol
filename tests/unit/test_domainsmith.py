"""Unit tests for the "import any field of knowledge" drafter (ui/domainsmith).

Covers the deterministic surface only — NO network / NO LLM:
  * `_fallback_draft` produces a valid, deterministic, in-scope draft from prose.
  * `to_domain` builds a `core.domains.Domain` (config, never belief).
  * `seed_claims_into` adds seed claims as NODES at the SKEPTICAL PRIOR (belief
    still moves only through the engine) and adds a provenance Source.
  * the ONTOLOGY GATE rejects a seed claim whose tag namespace is not one of the
    draft's declared entity/property types.
"""

from __future__ import annotations

import pytest

from cortesol.core.config import PRIOR_C_0
from cortesol.core.domains import Domain, get_active_domain, set_active_domain
from cortesol.core.kb import KB
from cortesol.core.mathx import logit
from cortesol.ui.domainsmith import (
    DomainDraft,
    SeedClaim,
    _fallback_draft,
    out_of_namespace_tags,
    seed_claims_into,
    to_domain,
    validate_draft,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_active_domain():
    """The active domain is process-global; never leak a switch into other tests."""
    original = get_active_domain()
    yield
    set_active_domain(original)


# --- _fallback_draft -------------------------------------------------------


def test_fallback_draft_is_deterministic():
    desc = "the effects of coffee consumption on sleep quality and mood"
    a = _fallback_draft(desc)
    b = _fallback_draft(desc)
    assert a.model_dump() == b.model_dump()


def test_fallback_draft_builds_an_in_scope_field_from_keywords():
    draft = _fallback_draft("the effects of coffee consumption on sleep quality")
    # keyed off the first significant token
    assert draft.name == "coffee"
    assert "coffee" in draft.entity_types
    assert draft.property_types  # at least one tracked property
    # every seed claim it invents is in-scope by construction (the gate is happy)
    assert out_of_namespace_tags(draft) == []
    assert draft.seed_claims
    assert validate_draft(draft) == []


def test_fallback_draft_handles_empty_description():
    draft = _fallback_draft("")
    assert draft.name  # never empty (never an empty namespace)
    assert draft.entity_types and draft.property_types


# --- to_domain -------------------------------------------------------------


def test_to_domain_builds_a_config_domain():
    draft = DomainDraft(
        name="Climate Science",
        label="Climate science",
        entity_types=["gas", "region"],
        property_types=["warming", "concentration"],
        out_of_scope_markers=["Astrology"],
        plausible_value_bounds={"warming_c": [0, 10], "concentration_ppm": [0, None]},
    )
    dom = to_domain(draft)
    assert isinstance(dom, Domain)
    assert dom.name == "climate_science"  # slugged
    assert dom.entity_types == ("gas", "region")
    assert dom.property_types == ("warming", "concentration")
    assert "gas" in dom.in_scope_namespaces and "warming" in dom.in_scope_namespaces
    # markers normalised to lowercase; bounds converted to tuples with None open sides
    assert dom.out_of_scope_markers == ("astrology",)
    assert dom.plausible_value_bounds["warming_c"] == (0.0, 10.0)
    assert dom.plausible_value_bounds["concentration_ppm"] == (0.0, None)


# --- seed_claims_into ------------------------------------------------------


def test_seed_claims_land_at_the_skeptical_prior():
    draft = DomainDraft(
        name="coffee",
        label="Coffee",
        entity_types=["coffee"],
        property_types=["sleep"],
        seed_claims=[
            SeedClaim(text="coffee affects sleep", tags=["coffee:coffee", "sleep:observed"]),
        ],
    )
    kb = KB()
    report = seed_claims_into(kb, draft, source_id="curator:coffee")

    assert len(report["seeded"]) == 1
    assert report["rejected"] == []
    (cid,) = report["seeded"]
    claim = kb.get_claim(cid)
    assert claim is not None
    # created structurally AT the prior — never a set confidence (PD2)
    assert claim.ell == pytest.approx(logit(PRIOR_C_0))
    assert claim.c == pytest.approx(PRIOR_C_0)
    assert claim.r == 0 and claim.s == 0
    assert claim.trajectory == []  # no evidence has moved it
    # a provenance Source was recorded
    assert kb.get_source("curator:coffee") is not None


def test_seed_claims_are_idempotent_on_the_same_text():
    draft = DomainDraft(
        name="coffee",
        label="Coffee",
        entity_types=["coffee"],
        property_types=["sleep"],
        seed_claims=[
            SeedClaim(text="coffee affects sleep", tags=["coffee:coffee", "sleep:observed"]),
            SeedClaim(text="coffee affects sleep", tags=["coffee:coffee", "sleep:observed"]),
        ],
    )
    kb = KB()
    report = seed_claims_into(kb, draft)
    assert len(kb.claims) == 1  # the duplicate text collapses to one node
    assert len(report["seeded"]) == 1


# --- the ontology gate -----------------------------------------------------


def test_ontology_gate_rejects_out_of_namespace_seed_tag():
    draft = DomainDraft(
        name="coffee",
        label="Coffee",
        entity_types=["coffee"],
        property_types=["sleep"],
        seed_claims=[
            SeedClaim(text="coffee affects sleep", tags=["coffee:coffee", "sleep:observed"]),
            # `peptide` is NOT one of this draft's declared namespaces -> must be rejected
            SeedClaim(text="coffee binds GLP1R", tags=["peptide:coffee", "sleep:observed"]),
        ],
    )
    # the gate flags the offending tag
    assert "peptide:coffee" in out_of_namespace_tags(draft)
    problems = validate_draft(draft)
    assert any("outside the declared ontology" in p for p in problems)

    # and seeding refuses to materialise the off-ontology claim
    kb = KB()
    report = seed_claims_into(kb, draft)
    assert len(report["seeded"]) == 1
    assert len(report["rejected"]) == 1
    assert report["rejected"][0]["bad_tags"] == ["peptide:coffee"]
    # only the in-scope claim exists in the graph
    texts = {c.text for c in kb.claims.values()}
    assert texts == {"coffee affects sleep"}


def test_seed_claim_with_no_tags_is_rejected():
    draft = DomainDraft(
        name="coffee",
        label="Coffee",
        entity_types=["coffee"],
        property_types=["sleep"],
        seed_claims=[SeedClaim(text="an untagged claim", tags=[])],
    )
    kb = KB()
    report = seed_claims_into(kb, draft)
    assert report["seeded"] == []
    assert len(report["rejected"]) == 1
