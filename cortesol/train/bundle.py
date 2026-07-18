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
RUNTIME_FILES = (
    "cortesol/__init__.py",
    "cortesol/core/__init__.py",
    "cortesol/core/config.py",
    "cortesol/core/context.py",
    "cortesol/core/domain.py",
    "cortesol/core/engine.py",
    "cortesol/core/kb.py",
    "cortesol/core/mathx.py",
    "cortesol/core/ops.py",
    "cortesol/core/propagate.py",
    "cortesol/core/results.py",
    "cortesol/core/schema.py",
    "cortesol/core/validator.py",
    "cortesol/ingest/__init__.py",
    "cortesol/ingest/quarantine.py",
    "cortesol/ingest/screen.py",
    "cortesol/pipeline.py",
    "cortesol/retrieval.py",
    "cortesol/sim/__init__.py",
    "cortesol/sim/events.py",
    "cortesol/sim/gold.py",
    "cortesol/sim/world.py",
    "cortesol/train/__init__.py",
    "cortesol/train/datasets.py",
    "cortesol/train/environment.py",
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "bundle-manifest.json"
    }


def _module_name(relative: str) -> tuple[str, bool]:
    path = Path(relative)
    if path.name == "__init__.py":
        return ".".join(path.parent.parts), True
    return ".".join((*path.parent.parts, path.stem)), False


def _write_runtime_bundle(path: Path, source_root: Path) -> None:
    """Embed exact package sources in a top-level helper accepted by `flash env push`."""
    sources: dict[str, str] = {}
    packages: list[str] = []
    for relative in RUNTIME_FILES:
        module, is_package = _module_name(relative)
        sources[module] = (source_root / relative).read_text(encoding="utf-8")
        if is_package:
            packages.append(module)
    source_json = json.dumps(sources, sort_keys=True, ensure_ascii=True)
    packages_json = json.dumps(sorted(packages), ensure_ascii=True)
    path.write_text(
        '"""Generated in-memory import bundle; do not edit."""\n'
        "from importlib import abc, util\n"
        "import sys\n"
        f"_SOURCES = {source_json}\n"
        f"_PACKAGES = frozenset({packages_json})\n"
        "class _RuntimeFinder(abc.MetaPathFinder, abc.Loader):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname not in _SOURCES:\n"
        "            return None\n"
        "        return util.spec_from_loader(fullname, self, is_package=fullname in _PACKAGES)\n"
        "    def create_module(self, spec):\n"
        "        return None\n"
        "    def exec_module(self, module):\n"
        "        name = module.__spec__.name\n"
        "        filename = f'<cortesol-runtime:{name}>'\n"
        "        module.__file__ = filename\n"
        "        exec(compile(_SOURCES[name], filename, 'exec'), module.__dict__)\n"
        "_FINDER = _RuntimeFinder()\n"
        "def install():\n"
        "    if not any(isinstance(item, _RuntimeFinder) for item in sys.meta_path):\n"
        "        sys.meta_path.insert(0, _FINDER)\n",
        encoding="utf-8",
    )


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
    _write_runtime_bundle(out / "cortesol_runtime_bundle.py", root)
    (out / "environment.py").write_text(
        "from cortesol_runtime_bundle import install as _install_runtime\n"
        "_install_runtime()\n"
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
