"""Training pipeline on Freesolo Flash (logical area B): SFT -> GRPO -> OPD.
The GRPO environment imports the REAL core engine+validator so the reward is
computed against the true belief state. Fine-Tuning Plan §Stage 1-3.
"""
