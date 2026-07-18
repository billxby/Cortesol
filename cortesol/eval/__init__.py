"""Eval (area C) — the proof both prizes ask for. Replay held-out streams, score
Brier/ECE/ASR against ground truth, and run the baselines that anchor the story:
gullible bot, stubborn bot, raw frontier LLM, engine+stock, engine+SFT, engine+GRPO.
Fine-Tuning Plan §eval-protocol.
"""

from __future__ import annotations

from .baselines import GullibleBot, StubbornBot
from .metrics import (
    asr,
    attack_success_rate,
    brier,
    ece,
    fraud_accepted_rate,
    max_confidence_shift,
)
from .redteam import attack_stream

__all__ = [
    "brier",
    "ece",
    "asr",
    "attack_success_rate",
    "fraud_accepted_rate",
    "max_confidence_shift",
    "GullibleBot",
    "StubbornBot",
    "attack_stream",
]
