"""Build the deterministic publishable Freesolo environment bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from ..core.ops import ops_json_schema

PUBLISHED_SPLITS = ("sft_smoke", "sft_train", "rl_train")
FIXED_MTIME = 946_684_800


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "bundle-manifest.json"
    }


def build_bundle(
    out_dir: str | Path,
    *,
    data_dir: str | Path,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    out = Path(out_dir)
    data = Path(data_dir)
    root = Path(source_root) if source_root else Path(__file__).parents[2]
    if out.resolve() in {root.resolve(), root.parent.resolve()}:
        raise ValueError("bundle output must not be the repository or its parent")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(
        root / "cortesol",
        out / "cortesol",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "dataset"),
    )
    dataset_out = out / "dataset"
    dataset_out.mkdir(parents=True)
    for split in PUBLISHED_SPLITS:
        src = data / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"missing generated split: {src}")
        shutil.copy2(src, dataset_out / f"{split}.jsonl")
    shutil.copy2(data / "rl_train.jsonl", dataset_out / "train.jsonl")
    shutil.copy2(root / "cortesol" / "train" / "TRAINING_CONTRACT.md", out / "TRAINING_CONTRACT.md")
    (out / "environment.py").write_text(
        "from cortesol.train.environment import BeliefUpdateEnv, load_environment\n"
        "__all__ = ['BeliefUpdateEnv', 'load_environment']\n",
        encoding="utf-8",
    )
    schema = ops_json_schema()
    (out / "ops.schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    public_manifest = {
        "dataset_version": source_manifest["dataset_version"],
        "contract_version": source_manifest["contract_version"],
        "git_commit": source_manifest["git_commit"],
        "files": {name: source_manifest["files"][name] for name in PUBLISHED_SPLITS},
        "sealed_data_included": False,
    }
    (out / "dataset-manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = _tree_manifest(out)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest: dict[str, object] = {"sha256": digest, "files": files}
    (out / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    forbidden = {"dev.jsonl", "final.jsonl", "security.jsonl"}
    present = {path.name for path in out.rglob("*") if path.is_file()}
    if present & forbidden:
        raise ValueError(f"sealed datasets leaked into bundle: {sorted(present & forbidden)}")
    for path in sorted(out.rglob("*")):
        os.utime(path, (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)
        if path.is_file():
            path.chmod(0o644)
    return manifest
