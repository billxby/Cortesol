from __future__ import annotations

import pytest
from pydantic import ValidationError

from cortesol.core.ops import ApplyEvidence, ProposedOps, ops_json_schema


def test_wire_schema_requires_every_operation_discriminator() -> None:
    schema = ops_json_schema()
    operation_defs = [
        definition
        for definition in schema["$defs"].values()
        if "op" in definition.get("properties", {})
    ]

    assert operation_defs
    assert all("op" in definition["required"] for definition in operation_defs)


def test_missing_wire_discriminator_fails_but_python_default_remains() -> None:
    with pytest.raises(ValidationError):
        ProposedOps.model_validate(
            {
                "ops": [
                    {
                        "claim_id": "P0_binding",
                        "direction": "+",
                        "strength": "moderate",
                        "evidence_id": "E0",
                    }
                ]
            }
        )

    operation = ApplyEvidence(
        claim_id="P0_binding",
        direction="+",
        strength="moderate",
        evidence_id="E0",
    )
    assert operation.op == "APPLY_EVIDENCE"
