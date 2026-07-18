"""Eval (area C) — the proof both prizes ask for. Replay held-out streams, score
Brier/ECE/ASR against ground truth, and run the baselines that anchor the story:
gullible bot, stubborn bot, raw frontier LLM, engine+stock, engine+SFT, engine+GRPO.
Fine-Tuning Plan §eval-protocol.
"""
