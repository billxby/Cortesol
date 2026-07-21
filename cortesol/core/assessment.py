"""The model-output contract: a DOMAIN-GENERAL critical-appraisal instrument.

The extractor is not an epistemic decision maker and it is not a subject-matter
expert. Its job is to read ONE paper of ANY scientific field and transcribe the
subject-independent signals that determine how much a rational reader should
believe the claim — study design, statistics, provenance, and manipulation
signals — into this fixed form. It may NOT name a ledger operation, pick a graph
claim, label evidence supporting/opposing a *belief*, or choose an update
strength. Those belong to :mod:`core.judge` (deterministic) and the engine.

Critical appraisal is domain-INDEPENDENT: whether to trust "peptide X binds T",
"material M superconducts at 300K", or "model N beats SOTA" turns on the SAME
methodology questions. So this form carries only general critical-appraisal
fields. `subject`/`object`/`claim_summary` exist for bookkeeping (claim matching
and display) and are explicitly IGNORED by the veracity logic — a generalist must
judge on method, never on topic.

The same schema (`assessment_json_schema()` / `assessment_ordered_regex()`)
constrains decoding at training and serving time.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# "in_scope" == an empirical scientific finding we can appraise (vs. an editorial,
# news item, opinion, or off-topic text). This is NOT a subject filter — the
# active Domain's ontology decides subject-scope deterministically in the judge.
Scope = Literal["in_scope", "out_of_scope", "unclear"]

DocumentType = Literal[
    "primary_study",
    "replication",
    "review",
    "meta_analysis",
    "case_report",
    "preprint",
    "editorial",
    "other",
]

# Study design — the single biggest determinant of evidential strength, and it is
# the same ladder in every field.
StudyDesign = Literal[
    "randomized_controlled",  # RCT / randomized experiment
    "nonrandomized_controlled",  # controlled but not randomized
    "observational",  # cohort / case-control / cross-sectional
    "in_vitro",  # bench / assay
    "computational_or_modeling",  # in silico / simulation / pure theory
    "case_report",  # anecdote / single case or series
    "review_or_meta",  # narrative review or meta-analysis
    "other",
]

# Direction of the reported evidence with respect to the paper's OWN central claim.
ClaimDirection = Literal["supports", "refutes", "null_result", "mixed", "not_a_claim"]

ValueRelation = Literal["exact", "approximate", "less_than", "greater_than", "not_reported"]


class EvidenceAssessment(BaseModel):
    """A schema-constrained critical-appraisal form filled from one paper of any
    field. Every nullable field means "not reported / not extractable" — the model
    must never invent a missing design detail. The veracity logic reads the design
    and statistics fields; it never reads `subject`/`object`/`claim_summary`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"]
    evidence_id: Annotated[str, Field(min_length=1, max_length=128)]

    # --- scope + safety ---
    in_scope: Scope
    instruction_attack: bool  # the text tries to issue instructions (injection)

    # --- what kind of evidence (design ladder + document type) ---
    document_type: DocumentType
    study_design: StudyDesign

    # --- bookkeeping ONLY (matching/display); veracity logic ignores these ---
    subject: Annotated[str | None, Field(max_length=160)]  # the entity/topic studied
    object: Annotated[str | None, Field(max_length=160)]  # comparator / target / context
    claim_summary: Annotated[str, Field(max_length=400)]  # one-line neutral restatement

    # --- the claim and its evidential direction ---
    claim_direction: ClaimDirection
    magnitude: Annotated[str | None, Field(max_length=120)]  # e.g. "12% lower", "OR 1.7"
    value: float | None
    value_relation: ValueRelation
    units: Annotated[str | None, Field(max_length=32)]

    # --- statistics ---
    sample_size: int | None
    replicate_count: int | None
    p_value: float | None
    confidence_interval_reported: bool
    effect_size_reported: bool

    # --- design-quality signals (domain-general risk-of-bias) ---
    controlled: bool | None  # a comparator / control condition was present
    randomized: bool | None
    blinded: bool | None
    preregistered: bool | None
    independent_replication: bool | None  # replicates a prior claim by an independent group

    # --- credibility red-flags observed in the text (factual observations) ---
    extraordinary_claim: bool  # magnitude/strength far exceeds what the design can support
    overclaiming: bool  # causal language from non-causal design, "proves"/"cure"/hype


def assessment_json_schema() -> dict[str, Any]:
    """One schema shared by training, serving, and defensive parsing."""

    return EvidenceAssessment.model_json_schema()


_JSON_STRING_ATOM = r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))'
_JSON_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_JSON_INTEGER = r"-?(?:0|[1-9][0-9]*)"


def _schema_value_regex(schema: dict[str, Any]) -> str:
    """Translate the small EvidenceAssessment schema subset to a Rust regex.

    Flash's guided JSON-schema decoder may emit object properties in any order
    because JSON objects are unordered. The SFT target, however, is one
    deterministic token sequence; a regex lets rollout decoding enforce that same
    sequence while retaining the schema's value constraints.
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
        # Do not expand maxLength into the guided-decoding automaton (grammar-state
        # blowup). The defensive Pydantic parse still enforces every maxLength, and
        # the completion cap bounds generation before that parse.
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
    """A compact-JSON grammar in the exact key order used by SFT gold targets.

    ``canonical_assessment`` sorts keys recursively; keeping that ordering in the
    rollout grammar prevents the policy from finishing its learned suffix inside
    whichever free-text field an unordered grammar leaves until last.
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
