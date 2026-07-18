"""The ground-truth peptide world (area B).

Latent truth values z_j and effect sizes theta_j over a peptide claim graph
(peptide binds/inhibits/stabilizes target, etc. — see core/domain.py). This is
the calibration target the KB never sees directly; only the eval harness reads it.

Reference: Fine-Tuning Plan §Stage 0.
Run `python -m cortesol.sim.world --demo` to print a toy stream (see Makefile `sim`).
"""

from __future__ import annotations


class World:
    """A seeded latent peptide world. Deterministic given a seed."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        ...  # TODO: sample the latent claim graph + truths + effect sizes


if __name__ == "__main__":
    ...  # TODO: --demo prints a short event stream
