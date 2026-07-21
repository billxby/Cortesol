"""Unit tests for the pluggable Domain abstraction (T0).

Locks two things the rest of build-week builds on: (1) the peptide path is the
default and unchanged, and (2) a non-peptide domain cleanly skips the
peptide-specific screen checks instead of mis-firing them.
"""

from __future__ import annotations

import pytest

from cortesol.core.domains import (
    PEPTIDES,
    Domain,
    get_active_domain,
    set_active_domain,
)
from cortesol.core.kb import KB
from cortesol.core.schema import Evidence
from cortesol.ingest.screen import screen

GENERIC = Domain(
    name="widgets",
    label="Widget manufacturing",
    entity_types=("widget", "factory"),
    property_types=("throughput", "defect_rate"),
)


@pytest.fixture(autouse=True)
def _restore_active_domain():
    """The active domain is process-global; never leak a switch into other tests."""
    original = get_active_domain()
    yield
    set_active_domain(original)


def test_default_active_domain_is_peptides():
    assert get_active_domain().name == "peptides"


def test_in_scope_namespaces_are_entity_plus_property_types():
    assert "peptide" in PEPTIDES.in_scope_namespaces
    assert "efficacy" in PEPTIDES.in_scope_namespaces
    assert "widget" in GENERIC.in_scope_namespaces
    assert "throughput" in GENERIC.in_scope_namespaces
    # a peptide namespace is NOT in scope for the widget domain
    assert "peptide" not in GENERIC.in_scope_namespaces


def test_set_active_domain_switches_and_registers():
    set_active_domain(GENERIC)
    assert get_active_domain().name == "widgets"
    # accepts a registered name too
    set_active_domain("peptides")
    assert get_active_domain().name == "peptides"


def test_generic_domain_skips_peptide_plausibility_flags():
    # A sub-picomolar Kd trips the peptide diffusion-limit flag...
    ev = Evidence(
        id="e1",
        source_id="s1",
        raw_text="a manufacturing report",
        fields={"metric": "Kd", "units": "nM", "value": 0.0001},
    )
    kb = KB()

    set_active_domain(PEPTIDES)
    assert "affinity_below_diffusion_limit" in screen(ev, kb)

    # ...but a domain with no such physical bound must not invent it.
    set_active_domain(GENERIC)
    assert "affinity_below_diffusion_limit" not in screen(ev, kb)


def test_generic_domain_uses_default_correlation_group():
    # lab x method x dataset grouping is domain-agnostic and available by default.
    group = GENERIC.correlation_group({"lab": "acme", "method": "line-3"})
    assert group == "acme|line-3|?"
