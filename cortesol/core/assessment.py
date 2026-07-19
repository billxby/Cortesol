"""The model-output contract: a neutral evidence intake form.

The extractor is not an epistemic decision maker.  It may transcribe what a
paper reports and which study-design features are present, but it cannot name a
ledger operation, select a graph claim, label evidence as supporting/opposing,
or choose an update strength.  Those decisions belong to :mod:`core.judge`.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Scope = Literal["peptide", "non_peptide", "unclear"]
DocumentType = Literal[
    "primary_research",
    "replication",
    "review",
    "meta_analysis",
    "case_report",
    "other",
]
StudyType = Literal[
    "binding",
    "in_vitro",
    "in_vivo",
    "clinical",
    "review",
    "other",
]
PropertyName = Literal[
    "binding_affinity",
    "potency",
    "selectivity",
    "serum_stability",
    "thermal_stability",
    "permeability",
    "solubility",
    "immunogenicity",
    "toxicity",
    "efficacy",
    "synthesis",
    "unknown",
]
Finding = Literal[
    "observed",
    "not_observed",
    "increased",
    "decreased",
    "no_difference",
    "mixed",
    "not_reported",
]
ValueRelation = Literal["exact", "approximately", "less_than", "greater_than", "not_reported"]


class EvidenceAssessment(BaseModel):
    """Schema-constrained form filled from one paper/report.

    Every nullable field means "not reported or not extractable".  Unknown is
    preserved as unknown; the model must not invent missing design details.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    evidence_id: Annotated[str, Field(min_length=1, max_length=128)]
    scope: Scope
    instruction_attack: bool
    document_type: DocumentType
    study_type: StudyType
    peptide: Annotated[str | None, Field(max_length=128)]
    target_or_indication: Annotated[str | None, Field(max_length=160)]
    property: PropertyName
    assay: Annotated[str | None, Field(max_length=128)]
    endpoint: Annotated[str | None, Field(max_length=128)]
    finding: Finding
    value: float | None
    value_relation: ValueRelation
    units: Annotated[str | None, Field(max_length=32)]
    sample_size: int | None
    replicate_count: int | None
    p_value: float | None
    randomized: bool | None
    blinded: bool | None
    controlled: bool | None
    preregistered: bool | None
    control_peptide: bool | None
    purity_pct: float | None


def assessment_json_schema() -> dict[str, Any]:
    """One schema shared by training, serving, and defensive parsing."""

    return EvidenceAssessment.model_json_schema()


_JSON_STRING_ATOM = r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))'
_JSON_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_JSON_INTEGER = r"-?(?:0|[1-9][0-9]*)"


def _schema_value_regex(schema: dict[str, Any]) -> str:
    """Translate the small EvidenceAssessment schema subset to a Rust regex.

    Flash's guided JSON-schema decoder is allowed to emit object properties in
    any order because JSON objects are unordered.  The SFT target, however, is
    one deterministic token sequence.  A regex lets rollout decoding enforce
    that same sequence while retaining the schema's value constraints.
    """

    if "anyOf" in schema:
        return "(?:" + "|".join(_schema_value_regex(item) for item in schema["anyOf"]) + ")"
    if "const" in schema:
        return re.escape(json.dumps(schema["const"], separators=(",", ":")))
    if "enum" in schema:
        return "(?:" + "|".join(
            re.escape(json.dumps(item, separators=(",", ":"))) for item in schema["enum"]
        ) + ")"

    value_type = schema.get("type")
    if value_type == "string":
        minimum = int(schema.get("minLength", 0))
        # Do not expand maxLength into the guided-decoding automaton. Repeating
        # the escaped-string alternation 32--160 times made a 24-turn GRPO
        # rollout spend minutes in grammar-state traversal at ~3% GPU use. The
        # defensive Pydantic parse still enforces every maxLength, while the
        # 256-token completion cap bounds generation before that parse.
        quantifier = "+" if minimum else "*"
        return f'"{_JSON_STRING_ATOM}{quantifier}"'
    if value_type == "number":
        return _JSON_NUMBER
    if value_type == "integer":
        return _JSON_INTEGER
    if value_type == "boolean":
        return "(?:true|false)"
    if value_type == "null":
        return "null"
    raise ValueError(f"unsupported assessment schema fragment: {schema}")


def assessment_ordered_regex() -> str:
    """A compact-JSON grammar in the exact order used by SFT gold targets.

    ``canonical_assessment`` sorts keys recursively.  Keeping that ordering in
    the rollout grammar prevents the policy from trying to finish its learned
    suffix inside whichever free-text property an unordered JSON grammar leaves
    until last.
    """

    schema = assessment_json_schema()
    properties = schema["properties"]
    required = set(schema["required"])
    if required != set(properties):
        raise ValueError("ordered assessment grammar requires every property")
    fields = [
        re.escape(json.dumps(name)) + ":" + _schema_value_regex(properties[name])
        for name in sorted(properties)
    ]
    return r"\{" + ",".join(fields) + r"\}"


def canonical_assessment(assessment: EvidenceAssessment) -> str:
    return json.dumps(
        assessment.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
