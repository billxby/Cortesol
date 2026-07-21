"""The peptides `Domain` — wraps the frozen peptide vocabulary (`core/domain.py`)
into a `Domain` instance. It references the existing constants/functions verbatim,
so the peptide path is byte-identical to before the Domain refactor.

Adding a new field of knowledge means writing a sibling of this file (or, in T4,
generating one from an LLM-drafted config) — never touching the engine.
"""

from __future__ import annotations

from .. import config as core_config
from .. import domain as pep
from .base import Domain

PEPTIDES = Domain(
    name="peptides",
    label="Peptides research",
    entity_types=pep.ENTITY_TYPES,
    property_types=pep.PROPERTY_TYPES,
    out_of_scope_examples=pep.OUT_OF_SCOPE_EXAMPLES,
    out_of_scope_markers=core_config.OUT_OF_SCOPE_MARKERS,
    evidence_field_keys=pep.EVIDENCE_FIELD_KEYS,
    assay_types=pep.ASSAY_TYPES,
    red_flag_phi_bump=pep.PEPTIDE_RED_FLAG_PHI_BUMP,
    knowledge=pep.PEPTIDE_KNOWLEDGE,
    min_plausible_kd_pm=pep.MIN_PLAUSIBLE_KD_PM,
    min_acceptable_purity_pct=pep.MIN_ACCEPTABLE_PURITY_PCT,
    correlation_group=pep.correlation_group,
)
