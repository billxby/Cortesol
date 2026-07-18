"""Build the SFT dataset (area B).

Rejection-sample K teacher completions per sim event; keep only those matching
gold (RFT). Keep rationales SHORT (small-model learnability gap). Emit Flash JSONL
rows {input, output, metadata} — every other top-level key is silently dropped, so
rubrics/gold go under metadata.

Reference: Fine-Tuning Plan §Stage 1.
"""

from __future__ import annotations


def build_sft_dataset(out_path: str, n_events: int) -> None:
    """Write a balanced-class SFT JSONL from the simulator + a teacher model."""
    ...  # TODO


if __name__ == "__main__":
    ...  # TODO: default build (see Makefile `train-sft`)
