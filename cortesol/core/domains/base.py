"""Domain — the pluggable field of knowledge the belief engine represents.

The belief ENGINE is domain-agnostic: `mathx / schema / ops / kb / context /
results / propagate / validator / engine` never hard-code a subject. Everything
subject-specific is packaged behind ONE object, a `Domain`: the tag ontology, the
scope boundary, the structured evidence-field schema, the domain-specific red
flags and physical-plausibility bounds, the n_eff correlation grouping, and the
curated seed knowledge.

`core/domain.py` still holds the peptide instance's raw data (a FROZEN CONTRACT
file); `domains/peptides.py` wraps it into a `Domain`. A new field of knowledge is
a new `Domain` instance — registered and activated (`domains.set_active_domain`)
with no change to the engine. This is the seam that makes "import any field" a
config problem, not an engine rewrite.

`core/` stays deterministic and LLM/network-free — a `Domain` is pure data plus
pure functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


def default_correlation_group(fields: Mapping[str, Any]) -> str:
    """The n_eff bucket for a report: lab x method x dataset, with journal/assay
    fallbacks so two venues reporting the same claim count as independent
    corroboration while repeats from one venue damp as echoes.

    Domain-agnostic — this is the same grouping the peptide domain has always
    used, promoted to the default for every field. A `Domain` may override it.
    (Confidence Math §3.)
    """
    lab = fields.get("lab") or fields.get("journal") or "?"
    method = fields.get("method") or fields.get("assay") or "?"
    dataset = fields.get("dataset") or "?"
    return "|".join(str(x) for x in (lab, method, dataset))


@dataclass(frozen=True)
class Domain:
    """One field of knowledge. See the module docstring.

    Only `name`, `label`, `entity_types`, and `property_types` are required; every
    other facet has a safe empty/None default so a minimal domain (e.g. an
    LLM-drafted draft in T4) is valid before its screening rules are filled in.
    A `None` plausibility bound means "this field has no such notion" — the screen
    simply skips that check, so a non-chemistry domain never trips peptide flags.
    """

    name: str
    label: str
    entity_types: tuple[str, ...]
    property_types: tuple[str, ...]
    out_of_scope_examples: tuple[str, ...] = ()
    out_of_scope_markers: tuple[str, ...] = ()
    evidence_field_keys: tuple[str, ...] = ()
    assay_types: tuple[str, ...] = ()
    # Which structured Evidence.fields augment the retrieval query. Defaults to the
    # historic peptide set so peptides stay byte-identical; other fields override.
    retrieval_query_fields: tuple[str, ...] = (
        "assay",
        "metric",
        "units",
        "organism",
        "cell_line",
    )
    # Domain-specific red-flag -> phi bumps, MERGED (not replacing) the generic
    # config.RED_FLAG_PHI_BUMP by the engine/screen.
    red_flag_phi_bump: Mapping[str, float] = field(default_factory=dict)
    # Curated seed CLAIM knowledge (never belief) keyed by canonical entity name.
    knowledge: Mapping[str, dict] = field(default_factory=dict)
    # Physical-sanity bounds for the deterministic screen; None -> skip the check.
    min_plausible_kd_pm: float | None = None
    min_acceptable_purity_pct: float | None = None
    # Generic physical-plausibility bounds, keyed by the reported `metric` name:
    # metric -> (min, max). A reported `value` outside its bound trips the generic
    # `value_physically_implausible` flag. `None` on either side = unbounded there.
    # This is the field-independent analogue of the peptide Kd diffusion-limit
    # check: a materials domain bounds conductivity, an ML domain caps accuracy at
    # 100%, etc. Peptides leaves this empty and keeps its bespoke Kd/purity checks,
    # so the peptide screen output is byte-identical.
    plausible_value_bounds: Mapping[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )
    # The n_eff bucketing function; a pure fn of Evidence.fields.
    correlation_group: Callable[[Mapping[str, Any]], str] = default_correlation_group

    @property
    def in_scope_namespaces(self) -> frozenset[str]:
        """A claim tag `<namespace>:<slug>` is in-scope iff its namespace is one of
        this domain's entity or property types."""
        return frozenset(tuple(self.entity_types) + tuple(self.property_types))
