"""Replay harness (area C). Run streams through the pipeline for each system and
print the eval table. Entry point for `make eval`.

Reference: Fine-Tuning Plan §eval-protocol. Held-out streams from unseen seeds +
a shifted event-class mix (OOD test), plus the red-team battery.

This is the ONE place sanctioned to read `sim_meta` (PD6): it reads `world_truth`
to build the scoring key and it seeds the claim graph from the latent `World`
(identities/text/tags only — never the truth `z` or the measured `value`, which
stay in the scoring key). Each system gets a FRESH KB seeded at the prior, is run
through `pipeline.replay_stream`, and is scored by `eval.metrics`.

Until `core/engine.py` + `pipeline.py` land, `replay_stream` is a no-op, so every
system sits at the prior and the table shows them tied — a real, honest baseline
("everything at c=PRIOR_C_0"). The numbers differentiate the moment belief starts
moving; nothing else here changes.
"""

from __future__ import annotations

from ..core.config import PRIOR_C_0
from ..core.kb import KB
from ..core.mathx import logit
from ..core.results import EventResult
from ..core.schema import Claim, RawEvent, Source
from ..sim.events import emit_stream
from ..sim.world import World
from . import metrics
from .baselines import GullibleBot, StubbornBot
from .redteam import attack_stream

# Held-out seed (unseen by any training/fixture) + a length that covers several
# full class rounds. Deterministic given the seed (sim uses a local RNG).
HELDOUT_SEED = 202
HELDOUT_LENGTH = 28
REDTEAM_SEED = 0


# --- harness plumbing -----------------------------------------------------


def _infer_tier(source_id: str) -> str:
    """Best-effort venue tier from a source id, so a seeded Source has a sensible
    cap for the validator. Placeholder heuristic — the live `adapters/cortex.py`
    (and `ingest/quarantine`) own real source tiering; this only has to be good
    enough for replay."""
    s = source_id.lower()
    if "spoof" in s:
        return "predatory"
    if "preprint" in s:
        return "preprint"
    if "nature" in s or "journal" in s or "reputable" in s:
        return "reputable"
    if "mail" in s or "chat" in s or "unknown" in s:
        return "unknown"
    return "unknown"


def seed_kb(world: World, events: list[RawEvent]) -> KB:
    """A fresh KB with the world's claims at the skeptical prior and a Source for
    every id the stream references. The sim never emits ADD_CLAIM (its gold moves
    belief on existing ids), so the harness must seed the claim graph."""
    kb = KB()
    for wc in world.claims:
        kb.add_claim(Claim(id=wc.id, text=wc.text, ontology_tags=list(wc.ontology_tags)))
    for e in events:
        if kb.get_source(e.source_id) is None:
            kb.add_source(Source.from_tier(e.source_id, _infer_tier(e.source_id)))
    return kb


def build_ground_truth(events: list[RawEvent]) -> dict[str, float]:
    """Merge every event's `sim_meta.world_truth` into one claim_id -> y map
    (y in {0.0, 1.0}). Only claims that received truth-labelled evidence appear."""
    gt: dict[str, float] = {}
    for e in events:
        if e.sim_meta is not None:
            for cid, z in e.sim_meta.world_truth.items():
                gt[cid] = float(z)
    return gt


def _replay(kb: KB, events: list[RawEvent], extractor: object) -> list[EventResult]:
    """Call the pipeline, tolerating the pre-engine stub state (returns None /
    raises NotImplementedError -> treat as 'no results', KB stays at the prior)."""
    from .. import pipeline  # imported late so eval is usable before pipeline lands

    try:
        results = pipeline.replay_stream(kb, events, extractor)  # type: ignore[arg-type]
    except NotImplementedError:
        return []
    return list(results) if results else []


def run_system(
    name: str, extractor: object, world: World, events: list[RawEvent]
) -> tuple[KB, list[EventResult]]:
    kb = seed_kb(world, events)
    return kb, _replay(kb, events, extractor)


def _belief_moved(kb: KB) -> bool:
    prior = logit(PRIOR_C_0)
    return any(abs(cl.ell - prior) > 1e-9 for cl in kb.claims.values())


# --- reporting ------------------------------------------------------------

_HEADER = f"{'system':<12} {'Brier↓':>8} {'ECE↓':>8} {'ASR↓':>8} {'fraud↓':>8} {'Δc_max↓':>9}"


def _row(name: str, kb: KB, results: list[EventResult], events: list[RawEvent],
         gt: dict[str, float]) -> str:
    return (
        f"{name:<12} "
        f"{metrics.brier(kb, gt):>8.3f} "
        f"{metrics.ece(kb, gt):>8.3f} "
        f"{metrics.asr(results, events):>8.3f} "
        f"{metrics.fraud_accepted_rate(results, events):>8.3f} "
        f"{metrics.max_confidence_shift(results):>9.3f}"
    )


def _table(
    title: str,
    world: World,
    events: list[RawEvent],
    systems: list[tuple[str, object]],
) -> bool:
    gt = build_ground_truth(events)
    print(f"\n== {title} ==")
    print(f"   ({len(events)} events, {len(gt)} truth-labelled claims)")
    print(_HEADER)
    print("-" * len(_HEADER))
    any_moved = False
    for name, extractor in systems:
        kb, results = run_system(name, extractor, world, events)
        any_moved = any_moved or _belief_moved(kb)
        print(_row(name, kb, results, events, gt))
    return any_moved


def main() -> None:
    """Replay held-out streams for every baseline + tuned system; print the table."""
    systems: list[tuple[str, object]] = [
        (GullibleBot.name, GullibleBot()),
        (StubbornBot.name, StubbornBot()),
    ]

    heldout_world = World(HELDOUT_SEED)
    heldout_stream = emit_stream(HELDOUT_SEED, HELDOUT_LENGTH)
    moved = _table("HELD-OUT STREAM", heldout_world, heldout_stream, systems)

    redteam_world = World(REDTEAM_SEED)
    redteam_events = attack_stream(REDTEAM_SEED)
    _table("RED-TEAM BATTERY", redteam_world, redteam_events, systems)

    if not moved:
        print(
            "\nNOTE: belief has not moved — core/engine.py + pipeline.py are still "
            f"stubs, so every system sits at the prior (c={PRIOR_C_0:.2f}). The harness "
            "is fully wired; the columns differentiate as soon as the engine lands."
        )
    print()


if __name__ == "__main__":
    main()
