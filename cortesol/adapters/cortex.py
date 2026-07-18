"""CORTEX / real-paper format adapter (area C) — the ONLY file that knows the
external stream format. Translates a real paper corpus (PubMed-style records) ->
our `RawEvent` (with sim_meta = None, since real events carry no gold), and seeds
a fresh KB from the public entities the corpus references.

If the upstream format differs from assumptions, ONLY this file changes.

Reference: Project Plan H0. `load_kb` (CORTEX belief-KB serialization) is left as a
documented kickoff-day stub; `load_stream` + `seed_kb_from_papers` handle the real
`data/papers/raw_papers.jsonl` corpus used by the live demo's "papers" mode.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.kb import KB
from ..core.schema import Claim, Edge, EdgeType, RawEvent, Source

# --- journal -> source tier (keyword heuristic over SOURCE_PRIORS tiers) -----
# Distrust routes to a lower reliability cap, never to fabricated disbelief. This
# mirrors eval/replay.py::_infer_tier but keyed on real journal names.
_TOP_JOURNALS = (
    "new england journal", "nejm", "nature", "science", "cell", "lancet", "jama",
)
_REPUTABLE = (
    "circulation", "diabetes care", "endocrin", "journal of the american",
    "bmj", "plos", "molecular", "pharmacolog", "drugs", "blood", "gut",
    "gastroenterology", "kidney", "hepatology", "clinical",
)
_PREPRINT = ("biorxiv", "medrxiv", "preprint", "arxiv", "ssrn", "research square")
_PREDATORY = ("omics", "predatory", "scirp", "hilaris")


def journal_to_tier(journal: str | None) -> str:
    """Best-effort venue tier for a journal name (keys core.config.SOURCE_PRIORS)."""
    j = (journal or "").lower()
    if any(k in j for k in _PREPRINT):
        return "preprint"
    if any(k in j for k in _PREDATORY):
        return "predatory"
    if any(k in j for k in _TOP_JOURNALS):
        return "top_journal"
    if any(k in j for k in _REPUTABLE):
        return "reputable"
    return "unknown"


def _journal_source_id(journal: str | None) -> str:
    """A stable source id from a journal name (the venue is the 'source')."""
    j = (journal or "unknown_venue").strip().lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in j).strip("_")
    return f"venue_{slug[:48]}" or "venue_unknown"


def load_stream(path: str | Path) -> list[RawEvent]:
    """Parse a real paper corpus (JSONL: pmid/title/abstract/journal/peptide/
    primary_target) into untrusted RawEvents. Measurements stay in the abstract
    prose for the extractor to read; there is no structured `fields` payload and
    no gold (`sim_meta = None`)."""
    events: list[RawEvent] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        title = rec.get("title", "") or ""
        abstract = rec.get("abstract", "") or ""
        events.append(
            RawEvent(
                id=f"pmid_{rec.get('pmid', i)}",
                t=len(events),
                source_id=_journal_source_id(rec.get("journal")),
                raw_text=(f"{title}\n\n{abstract}").strip(),
                fields={
                    "peptide": rec.get("peptide"),
                    "primary_target": rec.get("primary_target"),
                    "journal": rec.get("journal"),
                    "year": rec.get("year"),
                },
                sim_meta=None,
            )
        )
    return events


def seed_kb_from_papers(events: list[RawEvent]) -> KB:
    """A fresh KB seeded from the PUBLIC entities the corpus references — a Source
    per venue and one binding claim per unique (peptide, primary_target). This is
    the real-paper analogue of eval/replay.py::seed_kb (which seeds from `World`):
    it gives the extractor existing claims to APPLY onto, while novel propositions
    still arrive dynamically via ADD_CLAIM. No belief is set — claims start at the
    skeptical prior."""
    kb = KB()

    # one Source per venue, tiered by journal name
    for e in events:
        if kb.get_source(e.source_id) is None:
            tier = journal_to_tier(e.fields.get("journal"))
            kb.add_source(Source.from_tier(e.source_id, tier))

    # one binding claim per unique (peptide, target)
    binder_of: dict[str, str] = {}
    for e in events:
        pep = e.fields.get("peptide")
        tgt = e.fields.get("primary_target")
        if not pep or not tgt:
            continue
        cid = f"c_bind_{pep}_{tgt}"
        if cid not in kb.claims:
            kb.add_claim(
                Claim(
                    id=cid,
                    text=f"{pep} binds {tgt}",
                    ontology_tags=[f"peptide:{pep}", f"target:{tgt}", "binding_affinity:Kd"],
                )
            )
        binder_of.setdefault(str(pep), cid)

    # light typed topology: peptides that share a target support each other (same
    # program); derived only from public identities, never from any measurement.
    by_target: dict[str, list[str]] = {}
    for cid, cl in kb.claims.items():
        tgt = next((t.split(":", 1)[1] for t in cl.ontology_tags if t.startswith("target:")), None)
        if tgt:
            by_target.setdefault(tgt, []).append(cid)
    for cids in by_target.values():
        for a, b in zip(sorted(cids), sorted(cids)[1:]):
            eid = f"edge:{a}->{b}:supports"
            if eid not in kb.edges:
                kb.add_edge(Edge(id=eid, src=a, dst=b, type=EdgeType.SUPPORTS, weight=0.6))

    return kb


def load_kb(path: str) -> KB:
    """Parse CORTEX's belief KB serialization into our KB. Kickoff-day stub — the
    real CORTEX format is defined at the event; `load_stream` covers the paper
    corpus the demo uses today."""
    raise NotImplementedError("CORTEX KB format is defined at kickoff (Project Plan H0)")
