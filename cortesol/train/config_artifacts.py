"""Resolve checked-in Flash templates into immutable, schema-pinned TOML files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ..core.ops import ops_json_schema

CONFIG_DIR = Path(__file__).parent / "configs"
TEMPLATES = ("smoke_sft", "sft", "grpo", "opd", "grpo_opd")


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
    opd_adapter: str = "cortesol-opd",
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = ops_json_schema()
    schema_path = out / "ops.schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    adapters = {"grpo": sft_adapter, "opd": sft_adapter, "grpo_opd": opd_adapter}
    rendered: dict[str, Path] = {}
    for name in TEMPLATES:
        with (CONFIG_DIR / f"{name}.toml").open("rb") as handle:
            config = tomllib.load(handle)
        config["environment"]["id"] = environment_id
        if name in adapters:
            train = config["train"]
            train["init_from_adapter"] = adapters[name]
            train["structured_outputs"] = json.dumps(
                {"json": schema}, sort_keys=True, separators=(",", ":")
            )
            if "lora_rank" in train or "lora_alpha" in train:
                raise ValueError(f"warm-start config {name} must inherit LoRA metadata")
        path = out / f"{name}.toml"
        _write_toml(path, config)
        rendered[name] = path
    return rendered
