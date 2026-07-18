"""Foundation bootstrap — establish earned belief before the live demo (area C).

The demo should not open on a blank, maximally-skeptical graph: that makes for a
weak first impression and there is nothing for new results to *revise*. So we
pre-establish a foundation from a curated slice of the real corpus, then hold the
rest back to test robustness.

The one rule that makes this legitimate: **we do not set confidence.** We replay
the foundation papers through the *same* `pipeline.process_event` lifecycle every
other event takes (quarantine → retrieve → extract → screen → validate → engine →
propagate). Belief moves only through the engine, capped and provenance-carrying,
exactly as at serving time. Then we `kb.snapshot()` the result. Loading that
snapshot at demo start is caching the engine's *output*, not faking its *input* —
there is still no `SET_CONFIDENCE` anywhere. (See CLAUDE.md: "the LLM proposes;
the ledger disposes"; PD1.)

Selection: the "critical" papers per peptide are the highest-reliability ones
(venue tier, then corpus order). This is thesis-consistent — a top-tier source
legitimately earns a larger `logΛ` under the source cap, so the foundation ends
up confident *because the evidence warrants it*, not because we declared it.

Outputs (under `data/snapshots/`):
  * `foundation.json`     — the KB after replaying the foundation (KB.load-able).
  * `holdout.jsonl`       — the remaining papers, the robustness stream the UI runs.
  * `foundation_meta.json`— counts + which pmids are foundation vs holdout.

CLI:  python -m cortesol.bootstrap --per-peptide 3 [--extractor paper|flash|freesolo]
Make: make bootstrap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pipeline
from .adapters import cortex
from .core.kb import KB
from .core.mathx import sigmoid
from .core.schema import RawEvent

# Repo-root-relative locations (this file lives at cortesol/bootstrap.py).
_ROOT = Path(__file__).parents[1]
PAPERS_PATH = _ROOT / "data" / "papers" / "raw_papers.jsonl"
SNAPSHOT_DIR = _ROOT / "data" / "snapshots"
FOUNDATION_PATH = SNAPSHOT_DIR / "foundation.json"
HOLDOUT_PATH = SNAPSHOT_DIR / "holdout.jsonl"
META_PATH = SNAPSHOT_DIR / "foundation_meta.json"

# Lower rank = more reliable venue = preferred for the foundation. Mirrors the
# ordering implied by SOURCE_PRIORS caps without re-deriving the numbers here.
_TIER_RANK = {
    "top_journal": 0,
    "reputable": 1,
    "preprint": 2,
    "unknown": 3,
    "weak": 4,
    "predatory": 5,
}


def select_foundation(
    events: list[RawEvent], per_peptide: int = 3
) -> tuple[list[RawEvent], list[RawEvent]]:
    """Split the corpus into (foundation, holdout).

    Per peptide, take the `per_peptide` highest-reliability papers (venue tier,
    then original corpus order as a deterministic tiebreak). Everything else is
    holdout. Order within each list follows the original corpus order so replay is
    reproducible.
    """
    by_peptide: dict[str, list[tuple[int, RawEvent]]] = {}
    for i, e in enumerate(events):
        pep = e.fields.get("peptide")
        if not pep:
            continue
        by_peptide.setdefault(str(pep), []).append((i, e))

    foundation_ids: set[str] = set()
    for lst in by_peptide.values():
        ranked = sorted(
            lst,
            key=lambda ie: (
                _TIER_RANK.get(cortex.journal_to_tier(ie[1].fields.get("journal")), 9),
                ie[0],
            ),
        )
        for _, e in ranked[:per_peptide]:
            foundation_ids.add(e.id)

    foundation = [e for e in events if e.id in foundation_ids]
    holdout = [e for e in events if e.id not in foundation_ids]
    return foundation, holdout


def _make_extractor(kind: str):
    """Resolve an extractor for the bootstrap replay.

    Default is the deterministic, offline `PaperFakeExtractor`, so `make bootstrap`
    reproduces byte-for-byte with no network. `flash`/`freesolo` use the live model
    if its credentials are configured — the proposal source changes, the engine
    still disposes.
    """
    if kind in ("paper", "fake", "offline"):
        from .ingest.extract import PaperFakeExtractor

        return PaperFakeExtractor()
    if kind == "flash":
        from .eval.baselines import ModelBaseline

        return ModelBaseline("engine+flash")
    if kind in ("freesolo", "remote"):
        from .ingest.extract import FreesoloExtractor, _env, _load_dotenv

        _load_dotenv()
        run_id = _env("FREESOLO_RUN_ID")
        api_key = _env("FREESOLO_API_KEY")
        if not (run_id and api_key):
            raise RuntimeError("freesolo extractor needs FREESOLO_RUN_ID + FREESOLO_API_KEY")
        return FreesoloExtractor(run_id, api_key=api_key, api_url=_env("FLASH_API_URL"), timeout=60)
    raise ValueError(f"unknown extractor kind: {kind!r}")


def build_foundation(
    per_peptide: int = 3,
    extractor_kind: str = "paper",
    papers_path: Path = PAPERS_PATH,
    out_dir: Path = SNAPSHOT_DIR,
) -> dict:
    """Seed a KB from the full corpus, replay the foundation slice through the real
    lifecycle, snapshot the KB, and write the holdout stream + a manifest.

    Returns a summary dict (also written to `foundation_meta.json`).
    """
    events = cortex.load_stream(papers_path)
    foundation, holdout = select_foundation(events, per_peptide)

    # Seed from ALL events so every binding claim + venue source + shared-target
    # edge exists up front. The foundation replay moves belief only on the claims
    # its papers touch; the rest stay at the skeptical prior until holdout hits
    # them. No belief is set here — seeding is structural (claims start at PRIOR_C_0).
    kb = cortex.seed_kb_from_papers(events)

    extractor = _make_extractor(extractor_kind)
    results = pipeline.replay_stream(kb, foundation, extractor)

    out_dir.mkdir(parents=True, exist_ok=True)
    kb.snapshot(out_dir / "foundation.json")

    with (out_dir / "holdout.jsonl").open("w") as fh:
        for e in holdout:
            fh.write(json.dumps(e.model_dump(mode="json")) + "\n")

    # A confidence leaderboard makes the summary legible: what did the foundation
    # actually establish? Only claims the engine moved off the prior.
    moved = [c for c in kb.claims.values() if c.trajectory]
    moved.sort(key=lambda c: c.ell, reverse=True)
    top = [
        {"claim": c.id, "text": c.text, "c": round(sigmoid(c.ell), 3), "r": c.r, "s": c.s}
        for c in moved[:15]
    ]

    meta = {
        "per_peptide": per_peptide,
        "extractor": extractor_kind,
        "n_total": len(events),
        "n_foundation": len(foundation),
        "n_holdout": len(holdout),
        "n_claims": len(kb.claims),
        "n_claims_established": len(moved),
        "n_events_committed": len(results),
        "foundation_pmids": [e.id for e in foundation],
        "top_established_claims": top,
    }
    (out_dir / "foundation_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_foundation() -> tuple[KB, list[RawEvent]] | None:
    """Load the pre-built foundation KB + holdout stream, or None if not built yet.

    Used by the UI when `CORTESOL_FOUNDATION` is enabled. Returns the restored KB
    (claims carry their earned confidence and full audit trajectory) and the
    holdout papers as the stream to run live.
    """
    if not (FOUNDATION_PATH.exists() and HOLDOUT_PATH.exists()):
        return None
    kb = KB.load(FOUNDATION_PATH)
    events = [
        RawEvent(**json.loads(line))
        for line in HOLDOUT_PATH.read_text().splitlines()
        if line.strip()
    ]
    return kb, events


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the demo foundation snapshot.")
    ap.add_argument("--per-peptide", type=int, default=3, help="critical papers per peptide")
    ap.add_argument(
        "--extractor",
        default="paper",
        choices=["paper", "fake", "flash", "freesolo"],
        help="proposer for the foundation replay (default: offline deterministic)",
    )
    args = ap.parse_args()

    meta = build_foundation(per_peptide=args.per_peptide, extractor_kind=args.extractor)

    print(
        f"foundation: {meta['n_foundation']} papers -> "
        f"{meta['n_claims_established']}/{meta['n_claims']} claims established; "
        f"holdout: {meta['n_holdout']} papers"
    )
    print(f"  snapshot: {FOUNDATION_PATH.relative_to(_ROOT)}")
    print(f"  holdout : {HOLDOUT_PATH.relative_to(_ROOT)}")
    print("  top established claims (earned via the engine, not set):")
    for row in meta["top_established_claims"][:10]:
        print(f"    c={row['c']:.3f}  (r={row['r']} s={row['s']})  {row['text']}")


if __name__ == "__main__":
    main()
