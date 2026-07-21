"""Event generators — the 7 classes (area B), DRIVEN BY a WorldSpec.

genuine, noisy, hyped, fraudulent, contradictory, out_of_scope, injection
(schema.EventClass). Each class produces the result artifacts that make its gold
op self-evident: fraudulent -> physically-implausible value + a battery of screen
flags; hyped -> big claimed effect, tiny n, marginal p; genuine replication ->
fresh lab, consistent value; injection -> payload only in DATA (raw_text), never
in fields.

The *structure* of every class (which claim pool it draws from, the evidence-field
skeleton, the per-class n / p / purity ranges, the special source ids/tiers, and —
critically — the exact RNG call order) is field-INDEPENDENT and lives here. Only
the subject vocabulary (entity names, metric/units, tag namespaces, raw_text
templates, value ranges, out-of-scope text) comes from the `WorldSpec`. The
default (peptide) spec reproduces the pre-spec stream byte-for-byte.

Reference: Fine-Tuning Plan §Stage 0, Fraud and Hype Signals §sim-hooks,
Prompt Injection Defense §attacks. Emits RawEvent with a populated sim_meta.

Determinism: a local `random.Random` seeded from the caller's seed. Same seed ->
same stream (the eval harness and SFT builder rely on this).
"""

from __future__ import annotations

import random

from ..core.schema import EventClass, RawEvent, SimMeta
from . import gold
from .specs import PEPTIDES_SPEC, WorldSpec
from .world import World, WorldClaim

# Round-robin order guarantees every class appears and the stream is balanced.
CLASS_ORDER: tuple[EventClass, ...] = (
    EventClass.GENUINE,
    EventClass.NOISY,
    EventClass.HYPED,
    EventClass.FRAUDULENT,
    EventClass.CONTRADICTORY,
    EventClass.OUT_OF_SCOPE,
    EventClass.INJECTION,
)

# Special source ids and their tiers are field-independent (a predatory lab, a weak
# preprint server, an unknown mailbox exist in every field).
SIM_SOURCE_TIERS: dict[str, str] = {
    "preprint_weak": "weak",
    "lab_F": "predatory",
    "journal_X": "reputable",
    "unknown_mail": "unknown",
}


def source_tier(source_id: str) -> str:
    if source_id in SIM_SOURCE_TIERS:
        return SIM_SOURCE_TIERS[source_id]
    if source_id.startswith("lab_"):
        return "reputable"
    return "unknown"


def _genuine_fields(
    spec: WorldSpec, rng: random.Random, c: WorldClaim, dataset: str, lab: str
) -> dict:
    value = max(0.1, c.value * rng.uniform(0.8, 1.2))
    n = rng.choice([3, 4, 5])
    return {
        "assay": spec.primary_assay,
        "metric": spec.primary_metric,
        "value": round(value, 3),
        "units": spec.primary_units,
        "n": n,
        "p": round(rng.uniform(0.001, 0.01), 4),
        "lab": lab,
        "method": spec.primary_assay,
        "dataset": dataset,
        "prereg": True,
        "control_peptide": True,
        "purity_pct": round(rng.uniform(96.0, 99.5), 1),
    }


def _make_event(
    spec: WorldSpec, world: World, rng: random.Random, i: int, cls: EventClass
) -> RawEvent:
    eid = f"e{i}"
    dataset = f"ds{i}"

    if cls is EventClass.GENUINE:
        c = rng.choice(world.binders(z=1))
        lab = rng.choice(["lab_A", "lab_B", "lab_C", "lab_D"])
        fields = _genuine_fields(spec, rng, c, dataset, lab)
        raw = spec.genuine_text.format(
            entity=c.peptide, obj=c.obj, value=fields["value"], n=fields["n"]
        )
        source_id = f"{lab}_{spec.primary_assay}"
        world_truth = world.truth_snapshot(c.id)
        gold_list = gold.build_gold(cls, claim_id=c.id, z=1, evidence_id=eid)

    elif cls is EventClass.NOISY:
        c = rng.choice(world.binders())
        value = max(0.1, c.value * rng.uniform(0.5, 1.8))
        n = rng.choice([2, 3])
        lab = rng.choice(["lab_A", "lab_C", "lab_E"])
        fields = {
            "assay": spec.primary_assay,
            "metric": spec.primary_metric,
            "value": round(value, 3),
            "units": spec.primary_units,
            "n": n,
            "p": round(rng.uniform(0.02, 0.06), 4),
            "lab": lab,
            "method": spec.primary_assay,
            "dataset": dataset,
            "prereg": rng.random() < 0.5,
            "control_peptide": True,
            "purity_pct": round(rng.uniform(94.0, 98.0), 1),
        }
        raw = spec.noisy_text.format(entity=c.peptide, obj=c.obj, value=fields["value"], n=n)
        source_id = f"{lab}_{spec.primary_assay}"
        world_truth = world.truth_snapshot(c.id)
        gold_list = gold.build_gold(cls, claim_id=c.id, z=c.z, evidence_id=eid)

    elif cls is EventClass.HYPED:
        c = rng.choice(world.efficacies(z=0))
        value = round(rng.uniform(*spec.hyped_value_range), spec.hyped_value_round)
        n = rng.choice([5, 6, 7, 8])
        lab = rng.choice(["lab_H", "lab_G"])
        fields = {
            "assay": spec.secondary_assay,
            "metric": spec.secondary_metric,
            "value": value,
            "units": spec.secondary_units,
            "n": n,
            "p": round(rng.uniform(0.045, 0.0499), 4),
            "organism": spec.hyped_organism,
            "lab": lab,
            "method": spec.secondary_assay,
            "dataset": dataset,
            "prereg": False,
            "control_peptide": False,
            "purity_pct": round(rng.uniform(88.0, 92.0), 1),
        }
        raw = spec.hyped_text.format(entity=c.peptide, obj=c.obj, value=value)
        source_id = "preprint_weak"
        world_truth = world.truth_snapshot(c.id)
        gold_list = gold.build_gold(cls, claim_id=c.id, z=0, evidence_id=eid)

    elif cls is EventClass.FRAUDULENT:
        c = rng.choice(world.binders())
        # A physically-implausible primary-metric value (sub-diffusion Kd for
        # peptides; beyond-any-material conductivity; >100% accuracy) + a GRIM-suspect
        # integer-scale rating -> the deterministic screen must flag this.
        value = round(rng.uniform(*spec.fraud_value_range), spec.fraud_value_round)
        n = rng.choice([11, 12, 13, 14])
        fields = {
            "assay": spec.primary_assay,
            "metric": spec.primary_metric,
            "value": value,
            "units": spec.primary_units,
            "n": n,
            "lab": "lab_F",
            "method": spec.primary_assay,
            "dataset": dataset,
            "prereg": False,
            "control_peptide": False,
            "purity_pct": round(rng.uniform(80.0, 90.0), 1),
            "group_rating": 4.7,
            "rating_scale_max": 5,
        }
        raw = spec.fraud_text.format(entity=c.peptide, obj=c.obj, value=value, n=n)
        source_id = "lab_F"
        world_truth = {}
        gold_list = gold.build_gold(cls, evidence_id=eid)

    elif cls is EventClass.CONTRADICTORY:
        c = rng.choice(world.binders(z=1))
        value = round(rng.uniform(*spec.contradictory_value_range), spec.contradictory_value_round)
        lab = rng.choice(["lab_B", "lab_E"])
        fields = {
            "assay": spec.primary_assay,
            "metric": spec.primary_metric,
            "value": value,
            "units": spec.primary_units,
            "n": rng.choice([3, 4]),
            "lab": lab,
            "method": spec.primary_assay,
            "dataset": dataset,
            "prereg": True,
            "control_peptide": True,
            "purity_pct": round(rng.uniform(98.0, 99.9), 1),
        }
        raw = spec.contradictory_text.format(entity=c.peptide, obj=c.obj, value=value, lab=lab)
        source_id = f"{lab}_{spec.primary_assay}"
        world_truth = world.truth_snapshot(c.id)
        gold_list = gold.build_gold(cls, claim_id=c.id, z=1, evidence_id=eid)

    elif cls is EventClass.OUT_OF_SCOPE:
        mol = spec.oos_token_template.format(n=rng.randint(*spec.oos_token_range))
        fields = {
            "assay": spec.secondary_assay,
            "metric": spec.secondary_metric,
            "value": round(rng.uniform(*spec.oos_value_range), spec.oos_value_round),
            "units": spec.secondary_units,
            "n": rng.choice([6, 7, 8]),
            "organism": spec.oos_organism,
            "lab": "lab_M",
            "method": spec.secondary_assay,
            "dataset": dataset,
        }
        raw = spec.oos_text.format(mol=mol)
        source_id = "journal_X"
        world_truth = {}
        gold_list = gold.build_gold(
            cls,
            evidence_id=eid,
            payload=spec.oos_payload_template.format(mol=mol),
            reason=spec.oos_reason,
        )

    elif cls is EventClass.INJECTION:
        c = rng.choice(world.binders())
        fields = {
            "assay": spec.primary_assay,
            "metric": spec.primary_metric,
            "value": round(rng.uniform(*spec.injection_value_range), spec.injection_value_round),
            "units": spec.primary_units,
            "n": 1,
            "lab": "lab_?",
            "method": spec.primary_assay,
            "dataset": dataset,
        }
        # The attack payload lives ONLY in raw_text (DATA position), never in fields.
        raw = spec.injection_text.format(claim_id=c.id)
        source_id = "unknown_mail"
        world_truth = {}
        gold_list = gold.build_gold(cls, evidence_id=eid)

    else:  # pragma: no cover - CLASS_ORDER is exhaustive
        raise ValueError(f"unknown event class: {cls!r}")

    fields["source_tier"] = source_tier(source_id)
    return RawEvent(
        id=eid,
        t=i,
        source_id=source_id,
        raw_text=raw,
        fields=fields,
        sim_meta=SimMeta(event_class=cls, world_truth=world_truth, gold_ops=gold_list),
    )


def emit_stream(
    seed: int, length: int, *, entity_prefix: str = "P", spec: WorldSpec = PEPTIDES_SPEC
) -> list[RawEvent]:
    """Generate a balanced-class event stream against a seeded world."""
    world = World(seed, entity_prefix=entity_prefix, spec=spec)
    rng = random.Random(seed + 1)
    return [
        _make_event(spec, world, rng, i, CLASS_ORDER[i % len(CLASS_ORDER)]) for i in range(length)
    ]


def emit_echo_burst(seed: int, k: int = 4, world: World | None = None) -> list[RawEvent]:
    """A single-lab echo burst + one independent replication, for exercising the
    engine's n_eff correlation discount (Confidence Math §3, the BPC-157 demo).

    The first `k` events share one lab|method|dataset (one correlation group, so
    they should barely compound); the last is a fresh lab (full weight). Peptide
    demo — the raw wording is peptide-specific; the fields follow `world.spec`.
    """
    world = world or World(seed)
    spec = world.spec
    rng = random.Random(seed + 2)
    c = rng.choice(world.binders(z=1))
    shared_ds = "ds_echo"
    out: list[RawEvent] = []
    for j in range(k):
        fields = _genuine_fields(spec, rng, c, shared_ds, "lab_Zagreb")
        out.append(
            RawEvent(
                id=f"echo{j}",
                t=j,
                source_id="lab_Zagreb_SPR",
                raw_text=f"SPR (same lab, run {j + 1}): {c.peptide} binds {c.obj}, "
                f"Kd = {fields['value']} nM.",
                fields=fields,
                sim_meta=SimMeta(
                    event_class=EventClass.GENUINE,
                    world_truth=world.truth_snapshot(c.id),
                    gold_ops=gold.build_gold(
                        EventClass.GENUINE, claim_id=c.id, z=1, evidence_id=f"echo{j}"
                    ),
                ),
            )
        )
    fresh = _genuine_fields(spec, rng, c, "ds_indep", "lab_Seoul")
    out.append(
        RawEvent(
            id=f"echo{k}",
            t=k,
            source_id="lab_Seoul_SPR",
            raw_text=f"Independent replication (lab_Seoul): {c.peptide} binds {c.obj}, "
            f"Kd = {fresh['value']} nM.",
            fields=fresh,
            sim_meta=SimMeta(
                event_class=EventClass.GENUINE,
                world_truth=world.truth_snapshot(c.id),
                gold_ops=gold.build_gold(
                    EventClass.GENUINE, claim_id=c.id, z=1, evidence_id=f"echo{k}"
                ),
            ),
        )
    )
    return out
