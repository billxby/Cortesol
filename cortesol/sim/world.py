"""The ground-truth world (area B) — now DRIVEN BY a WorldSpec.

Latent truth values z_j and effect sizes theta_j over a claim graph. For the
default (peptide) spec this is exactly the historical peptide world: peptide
binds/inhibits target, is efficacious in an indication, etc. — see
core/domain.py. Swap the spec (sim/specs.py) and the same code renders a
materials or ML-benchmark world instead; the belief engine never changes. This is
the calibration target the KB never sees directly; only the eval harness reads it.

Deterministic given a seed (a local `random.Random(seed)` — never the global RNG
or wall-clock), so the same seed always yields the same world and the same stream.
The peptide spec is byte-identical to the pre-spec simulator (verified in tests).

Reference: Fine-Tuning Plan §Stage 0.
Run `python -m cortesol.sim.world --demo` to print a toy stream (see Makefile `sim`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .specs import PEPTIDES_SPEC, WorldSpec


@dataclass
class WorldClaim:
    """One latent proposition. `z` is truth (1/0); `value` is the true measured
    quantity a genuine report would recover (Kd in nM for binders, effect % for
    efficacy — or the spec's metric for another field). Generators add measurement
    noise on top of this. `peptide` holds the entity name (kept for backwards-compat
    with peptide callers; it is simply "the entity" for any field)."""

    id: str
    text: str
    ontology_tags: list[str]
    kind: str  # spec.primary_kind ("binding") | spec.secondary_kind ("efficacy")
    peptide: str  # the entity name
    obj: str  # target (primary) or indication (secondary)
    z: int
    value: float


@dataclass
class World:
    """A seeded latent world. Deterministic given a seed. Defaults to the peptide
    spec so `World(seed)` / `World(seed, entity_prefix="TR")` behave exactly as
    before; pass `spec=` to render another field of knowledge."""

    seed: int
    entity_prefix: str = "P"
    spec: WorldSpec = PEPTIDES_SPEC
    claims: list[WorldClaim] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        spec = self.spec
        prefix = self.entity_prefix
        # Primary claims: one per entity, truth alternating so a mix of true and
        # false is guaranteed regardless of seed. (RNG: one uniform per entity.)
        for k in range(1, spec.num_entities + 1):
            entity = f"{prefix}{k}"
            obj = spec.primary_objects[k % len(spec.primary_objects)]
            z = spec.primary_z(k)
            value = (
                rng.uniform(*spec.primary_true_range)
                if z
                else rng.uniform(*spec.primary_false_range)
            )
            self.claims.append(
                WorldClaim(
                    id=spec.primary_id_template.format(entity=entity, obj=obj),
                    text=spec.primary_text_template.format(entity=entity, obj=obj),
                    ontology_tags=[
                        t.format(entity=entity, obj=obj) for t in spec.primary_tag_templates
                    ],
                    kind=spec.primary_kind,
                    peptide=entity,
                    obj=obj,
                    z=z,
                    value=round(value, spec.primary_round),
                )
            )
        # Secondary claims for a fixed subset (so the hyped class always has a false
        # claim to draw on). (RNG: one uniform per member, AFTER all primary draws.)
        for k, z in spec.secondary_members:
            entity = f"{prefix}{k}"
            obj = spec.secondary_objects[k % len(spec.secondary_objects)]
            value = (
                rng.uniform(*spec.secondary_true_range)
                if z
                else rng.uniform(*spec.secondary_false_range)
            )
            self.claims.append(
                WorldClaim(
                    id=spec.secondary_id_template.format(entity=entity, obj=obj),
                    text=spec.secondary_text_template.format(entity=entity, obj=obj),
                    ontology_tags=[
                        t.format(entity=entity, obj=obj) for t in spec.secondary_tag_templates
                    ],
                    kind=spec.secondary_kind,
                    peptide=entity,
                    obj=obj,
                    z=z,
                    value=round(value, spec.secondary_round),
                )
            )

    # -- accessors the event generator uses (names kept for backwards-compat) --

    def binders(self, z: int | None = None) -> list[WorldClaim]:
        """Primary-kind claims (peptide: binders)."""
        return [
            c
            for c in self.claims
            if c.kind == self.spec.primary_kind and (z is None or c.z == z)
        ]

    def efficacies(self, z: int | None = None) -> list[WorldClaim]:
        """Secondary-kind claims (peptide: efficacies)."""
        return [
            c
            for c in self.claims
            if c.kind == self.spec.secondary_kind and (z is None or c.z == z)
        ]

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
    ap.add_argument("--domain", default="peptides", help="spec name (see sim/specs.py SPECS)")
    args = ap.parse_args()

    from .specs import SPECS

    spec = SPECS[args.domain]
    for event in emit_stream(args.seed, args.length, entity_prefix=spec.entity_prefix, spec=spec):
        print(event.model_dump_json())
