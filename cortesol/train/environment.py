"""The GRPO environment — the centerpiece training signal (area B).

Flash EnvironmentMultiTurn. One episode = one stream of ~15-30 sim events applied
through the REAL validator + engine (import cortesol.core). Terminal reward =
negative Brier of the final KB confidences vs. ground truth (a strictly proper
scoring rule -> can't be gamed by over/under-confidence). Small dense shaping:
schema-valid, provenance cited, correct REJECT on injection, correct FLAG_OOD.

Reference: Fine-Tuning Plan §Stage 2 (+ reward-hacking watchlist).
"""

from __future__ import annotations


class BeliefUpdateEnv:
    """EnvironmentMultiTurn: start_episode / step_episode / score_episode."""

    def score_episode(self) -> float:
        """R = -mean((c_i - y_i)^2) + capped shaping. TODO."""
        ...  # TODO
