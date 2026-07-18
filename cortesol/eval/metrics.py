"""Metrics (area C). All computed against the sim's ground truth (sim_meta).

Brier = mean((c_i - y_i)^2)  [also the GRPO reward];  ECE = calibration error;
ASR = injection attack success rate (any op exceeding gold under attack);
plus fraud-accepted rate, replication response, OOD-flag F1.
Reference: Fine-Tuning Plan §eval, Prompt Injection Defense §metrics.
"""

from __future__ import annotations

from ..core.kb import KB


def brier(kb: KB, ground_truth: dict[str, float]) -> float:
    """Mean squared error of claim confidences vs. truth. Lower is better."""
    ...  # TODO


def ece(kb: KB, ground_truth: dict[str, float], bins: int = 10) -> float:
    """Expected calibration error."""
    ...  # TODO
