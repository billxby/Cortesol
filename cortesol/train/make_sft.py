"""Build the SFT dataset (area B).

Rejection-sample K teacher completions per sim event; keep only those matching
gold (RFT). Keep rationales SHORT (small-model learnability gap). Emit Flash JSONL
rows {input, output, metadata} — every other top-level key is silently dropped, so
rubrics/gold go under metadata.

Reference: Fine-Tuning Plan §Stage 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import build_all


def build_sft_dataset(out_path: str, n_events: int = 2_800) -> None:
    """Write verified deterministic-gold SFT JSONL.

    ``n_events`` is retained for compatibility and must match the frozen
    production profile; use :func:`datasets.build_all` for other artifacts.
    """
    if n_events != 2_800:
        raise ValueError("the production profile is frozen at 2,800 examples")
    out = Path(out_path)
    manifest = build_all(out.parent)
    generated = out.parent / manifest["files"]["sft_train"]["path"]
    if generated != out:
        out.write_bytes(generated.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build all Cortesol training datasets")
    parser.add_argument("--out", default="runs/training/data")
    args = parser.parse_args()
    result = build_all(args.out)
    print(Path(args.out) / "manifest.json")
    print(json.dumps(result["files"], indent=2, sort_keys=True))
