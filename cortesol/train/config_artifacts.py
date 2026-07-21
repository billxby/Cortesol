"""Resolve checked-in Flash templates into immutable, schema-pinned TOML files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ..core.assessment import assessment_json_schema
from ..core.ops import ops_json_schema

CONFIG_DIR = Path(__file__).parent / "configs"
TEMPLATES = ("smoke_sft", "sft", "grpo", "opd", "grpo_opd")
# The SFT-ONLY appraisal ladder. The generalist student emits EvidenceAssessment,
# not ops, so there is no GRPO/OPD stage; every appraisal config gets the appraisal
# grammar pinned as structured_outputs (unlike the ops SFT, which learns the grammar).
APPRAISAL_TEMPLATES = ("appraisal_smoke", "appraisal_sft")

# The two structured-output targets a Flash run can be pinned to. `ops` (default)
# is the frozen ops-proposal grammar; `appraisal` is the domain-general critical-
# appraisal form (`core/assessment.EvidenceAssessment`) for the generalist model.
_TARGETS = {
    "ops": ("ops.schema.json", ops_json_schema),
    "appraisal": ("assessment.schema.json", assessment_json_schema),
}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def _dump_table(data: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[str]:
    lines: list[str] = []
    scalars = [(key, value) for key, value in data.items() if not isinstance(value, dict)]
    tables = [(key, value) for key, value in data.items() if isinstance(value, dict)]
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in scalars)
    for key, table in tables:
        if lines:
            lines.append("")
        lines.extend(_dump_table(table, (*prefix, key)))
    return lines


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.write_text("\n".join(_dump_table(data)).rstrip() + "\n", encoding="utf-8")


def render_configs(
    out_dir: str | Path,
    *,
    environment_id: str,
    sft_adapter: str = "cortesol-sft",
    grpo_adapter: str | None = None,
    opd_adapter: str = "cortesol-opd",
    target: str = "ops",
) -> dict[str, Path]:
    """Resolve the checked-in Flash templates into schema-pinned TOML files.

    ``target`` selects the structured-output grammar: ``"ops"`` (default, the frozen
    ops-proposal schema — byte-identical to before) or ``"appraisal"`` (the
    domain-general ``EvidenceAssessment`` form for the generalist appraisal model).
    """
    if target not in _TARGETS:
        raise ValueError(f"unknown structured-output target {target!r}; known: {sorted(_TARGETS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema_filename, schema_fn = _TARGETS[target]
    schema = schema_fn()
    schema_path = out / schema_filename
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    structured_outputs = json.dumps({"json": schema}, sort_keys=True, separators=(",", ":"))
    if target == "appraisal":
        # SFT-only: Flash 1.0 forbids train.structured_outputs on SFT (it trains on
        # dataset completions and never generates), so the appraisal grammar is pinned
        # only as the emitted assessment.schema.json artifact above — the gold rows
        # already match assessment_ordered_regex and serving re-imposes the schema.
        return _render_appraisal(out, environment_id)
    adapters = {
        "grpo": sft_adapter,
        "opd": grpo_adapter or sft_adapter,
        "grpo_opd": opd_adapter,
    }
    rendered: dict[str, Path] = {}
    for name in TEMPLATES:
        with (CONFIG_DIR / f"{name}.toml").open("rb") as handle:
            config = tomllib.load(handle)
        config["environment"]["id"] = environment_id
        train = config["train"]
        if name in adapters:
            train["init_from_adapter"] = adapters[name]
            train["structured_outputs"] = structured_outputs
            if "lora_rank" in train or "lora_alpha" in train:
                raise ValueError(f"warm-start config {name} must inherit LoRA metadata")
        path = out / f"{name}.toml"
        _write_toml(path, config)
        rendered[name] = path
    return rendered


def _render_appraisal(out: Path, environment_id: str) -> dict[str, Path]:
    """Resolve the SFT-only appraisal templates by stamping the published environment
    id (there is no ops warm-start / GRPO / OPD here). The appraisal grammar rides
    along as the emitted ``assessment.schema.json`` and as the gold completions'
    ordered form; Flash rejects ``structured_outputs`` on an SFT run."""
    rendered: dict[str, Path] = {}
    for name in APPRAISAL_TEMPLATES:
        with (CONFIG_DIR / f"{name}.toml").open("rb") as handle:
            config = tomllib.load(handle)
        config["environment"]["id"] = environment_id
        path = out / f"{name}.toml"
        _write_toml(path, config)
        rendered[name] = path
    return rendered


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ap = sub.add_parser(
        "appraisal", help="render the SFT-only appraisal configs with the appraisal grammar pinned"
    )
    ap.add_argument(
        "--environment-id",
        required=True,
        help="the published environment id from `flash env push` (owner/name)",
    )
    ap.add_argument("--out", default="runs/appraisal/configs")
    args = parser.parse_args()
    if args.command == "appraisal":
        paths = render_configs(args.out, environment_id=args.environment_id, target="appraisal")
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
