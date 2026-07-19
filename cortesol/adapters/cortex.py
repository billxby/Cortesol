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

from ..core.domain import (
    binding_claim_id,
    peptide_facts,
    property_claim_id,
)
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
    """Best-effort venue tier for a journal name (keys core.config.SOURCE_PRIORS).

    A recognised journal in a real PubMed corpus is peer-reviewed, so the DEFAULT is
    `reputable`, not `unknown` — "unknown" is epistemically as weak as knowing nothing
    and never belongs on a published paper. Distrust is still expressed by the lower
    caps of `preprint` / `predatory`; it is just never the fallback. (This mirrors
    `ingest/fetch_papers.source_tier`, which also defaults published venues to
    reputable.)"""
    j = (journal or "").lower()
    if any(k in j for k in _PREPRINT):
        return "preprint"
    if any(k in j for k in _PREDATORY):
        return "predatory"
    if any(k in j for k in _TOP_JOURNALS):
        return "top_journal"
    return "reputable"


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
        authors = rec.get("authors") or []
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
                    # display metadata (data, not instructions) for the browse view
                    "title": title,
                    "authors": authors[:6],
                    "pmid": rec.get("pmid"),
                    "doi": rec.get("doi"),
                },
                sim_meta=None,
            )
        )
    return events


def _resolve_target(peptide: str, event_target: str | None) -> str | None:
    """The grounded primary target for a peptide: curated knowledge wins, then a real
    target off the event, and an `unknown`/empty target resolves to None — we never
    seed a "binds unknown" claim (it is noise, not knowledge)."""
    facts = peptide_facts(peptide)
    if facts is not None:
        return facts.get("target")  # may legitimately be None (no receptor)
    tgt = (event_target or "").strip()
    if tgt.lower() in ("", "unknown", "none", "n/a", "null"):
        return None
    return tgt


def merge_papers_into_kb(kb: KB, events: list[RawEvent]) -> KB:
    """Add the PUBLIC entities a batch of papers references into an EXISTING KB — a
    Source per venue, a binding claim per unique (peptide, target), the peptide's
    curated GROUNDED property claims (efficacy / selectivity / stability from
    `domain.PEPTIDE_KNOWLEDGE`), and typed edges (shared-target SUPPORTS + each
    property DEPENDS_ON its binding claim). Idempotent: existing sources/claims/edges
    are left untouched, so this can be called repeatedly (e.g. as new papers arrive
    from the Research chat) and safely re-seeds only what is missing.

    No belief is set — every new claim starts at the skeptical prior, and confidence
    still moves only through the engine when the papers are later processed. Seeding
    a claim NODE just gives the extractor something grounded to APPLY evidence onto;
    a seeded claim that never receives evidence stays hidden (the UI only reveals
    claims a committed result has touched), so this widens the graph, not the noise."""
    # one Source per venue, tiered by journal name
    for e in events:
        if kb.get_source(e.source_id) is None:
            tier = journal_to_tier(e.fields.get("journal"))
            kb.add_source(Source.from_tier(e.source_id, tier))

    # per peptide: the binding claim (when a real target exists) + grounded properties
    for e in events:
        pep = (e.fields.get("peptide") or "").strip()
        if not pep:
            continue
        tgt = _resolve_target(pep, e.fields.get("primary_target"))
        binding_id = None
        if tgt:
            binding_id = binding_claim_id(pep, tgt)
            if binding_id not in kb.claims:
                kb.add_claim(
                    Claim(
                        id=binding_id,
                        text=f"{pep} binds {tgt}",
                        ontology_tags=[f"peptide:{pep}", f"target:{tgt}", "binding_affinity:Kd"],
                    )
                )

        facts = peptide_facts(pep)
        if not facts:
            continue
        for namespace, slug, text, _cues in facts["claims"]:
            cid = property_claim_id(namespace, pep, slug)
            if cid not in kb.claims:
                kb.add_claim(
                    Claim(
                        id=cid,
                        text=text,
                        ontology_tags=[f"peptide:{pep}", f"{namespace}:{slug}"],
                    )
                )
            # a property of the peptide DEPENDS_ON it actually binding its target —
            # the load-bearing relation the retraction cascade rides (matches the sim
            # topology in ui/app.py::_seed_demo_edges).
            if binding_id:
                edge_id = f"edge:{binding_id}->{cid}:depends_on"
                if edge_id not in kb.edges:
                    kb.add_edge(
                        Edge(
                            id=edge_id, src=binding_id, dst=cid,
                            type=EdgeType.DEPENDS_ON, weight=0.6,
                        )
                    )

    # light typed topology: peptides that share a target support each other (same
    # program); derived only from public identities, never from any measurement.
    by_target: dict[str, list[str]] = {}
    for cid, cl in kb.claims.items():
        tgt = next((t.split(":", 1)[1] for t in cl.ontology_tags if t.startswith("target:")), None)
        if tgt:
            by_target.setdefault(tgt, []).append(cid)
    for cids in by_target.values():
        for a, b in zip(sorted(cids), sorted(cids)[1:], strict=False):
            eid = f"edge:{a}->{b}:supports"
            if eid not in kb.edges:
                kb.add_edge(Edge(id=eid, src=a, dst=b, type=EdgeType.SUPPORTS, weight=0.6))

    return kb


def seed_kb_from_papers(events: list[RawEvent]) -> KB:
    """A fresh KB seeded from the PUBLIC entities the corpus references — a Source
    per venue and one binding claim per unique (peptide, primary_target). This is
    the real-paper analogue of eval/replay.py::seed_kb (which seeds from `World`):
    it gives the extractor existing claims to APPLY onto, while novel propositions
    still arrive dynamically via ADD_CLAIM. No belief is set — claims start at the
    skeptical prior."""
    return merge_papers_into_kb(KB(), events)


def load_kb(path: str) -> KB:
    """Parse CORTEX's belief KB serialization into our KB. Kickoff-day stub — the
    real CORTEX format is defined at the event; `load_stream` covers the paper
    corpus the demo uses today."""
    raise NotImplementedError("CORTEX KB format is defined at kickoff (Project Plan H0)")
