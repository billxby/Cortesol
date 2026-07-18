"""The ground-truth peptide world (area B).

Latent truth values z_j and effect sizes theta_j over a peptide claim graph
(peptide binds/inhibits/stabilizes target, etc. — see core/domain.py). This is
the calibration target the KB never sees directly; only the eval harness reads it.

Deterministic given a seed (a local `random.Random(seed)` — never the global RNG
or wall-clock), so the same seed always yields the same world and the same stream.

Reference: Fine-Tuning Plan §Stage 0.
Run `python -m cortesol.sim.world --demo` to print a toy stream (see Makefile `sim`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Small symbolic vocabulary. The engine is domain-agnostic; these are just labels
# that satisfy core.domain's in-scope rule (>=1 peptide entity AND >=1 property).
_TARGETS = ("GLP1R", "MC4R", "NPY2R", "GLP2R", "integrin_avb3", "CXCR4")
_INDICATIONS = ("obesity", "diabetes", "inflammation")
_NUM_PEPTIDES = 6


@dataclass
class WorldClaim:
    """One latent proposition. `z` is truth (1/0); `value` is the true measured
    quantity a genuine report would recover (Kd in nM for binders, effect % for
    efficacy). Generators add measurement noise on top of this."""

    id: str
    text: str
    ontology_tags: list[str]
    kind: str  # "binding" | "efficacy"
    peptide: str
    obj: str  # target (binding) or indication (efficacy)
    z: int
    value: float


@dataclass
class World:
    """A seeded latent peptide world. Deterministic given a seed."""

    seed: int
    claims: list[WorldClaim] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        # Binding claims: one per peptide, truth alternating so a mix of true and
        # false binders is guaranteed regardless of seed.
        for k in range(1, _NUM_PEPTIDES + 1):
            pep = f"P{k}"
            target = _TARGETS[k % len(_TARGETS)]
            z = 1 if k % 2 == 1 else 0
            kd = rng.uniform(1.0, 40.0) if z else rng.uniform(6000.0, 40000.0)
            self.claims.append(
                WorldClaim(
                    id=f"c_bind_{pep}_{target}",
                    text=f"{pep} binds {target}",
                    ontology_tags=[f"peptide:{pep}", f"target:{target}", "binding_affinity:Kd"],
                    kind="binding",
                    peptide=pep,
                    obj=target,
                    z=z,
                    value=round(kd, 4),
                )
            )
        # Efficacy claims for a subset. P1 is truly efficacious; P3/P5 are not, so
        # there is always a false efficacy claim for the `hyped` class to draw on.
        for k, z in ((1, 1), (3, 0), (5, 0)):
            pep = f"P{k}"
            indication = _INDICATIONS[k % len(_INDICATIONS)]
            effect = rng.uniform(45.0, 90.0) if z else rng.uniform(0.0, 15.0)
            self.claims.append(
                WorldClaim(
                    id=f"c_efficacy_{pep}_{indication}",
                    text=f"{pep} is efficacious in {indication}",
                    ontology_tags=[f"peptide:{pep}", f"efficacy:{indication}"],
                    kind="efficacy",
                    peptide=pep,
                    obj=indication,
                    z=z,
                    value=round(effect, 2),
                )
            )

    # -- accessors the event generator uses --

    def binders(self, z: int | None = None) -> list[WorldClaim]:
        return [c for c in self.claims if c.kind == "binding" and (z is None or c.z == z)]

    def efficacies(self, z: int | None = None) -> list[WorldClaim]:
        return [c for c in self.claims if c.kind == "efficacy" and (z is None or c.z == z)]

    def truth_snapshot(self, claim_id: str) -> dict[str, int]:
        for c in self.claims:
            if c.id == claim_id:
                return {c.id: c.z}
        return {}


if __name__ == "__main__":
    import argparse

    from .events import emit_stream

    ap = argparse.ArgumentParser(description="Print a toy simulator stream as JSONL.")
    ap.add_argument("--demo", action="store_true", help="print a short demo stream")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=int, default=14)
    args = ap.parse_args()

    for event in emit_stream(args.seed, args.length):
        print(event.model_dump_json())
