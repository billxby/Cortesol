"""Retrieval — build the bounded Context the extractor reads (area C).

Embed the incoming event -> top-k relevant claims + their 1-hop neighborhood +
source records (config.RETRIEVE_TOP_K). Must fit CONTEXT_TOKEN_BUDGET. Both the
live pipeline and the GRPO environment call this, so keep it dependency-light; a
trivial tag/recency version works before embeddings land.

Reference: System Architecture §lifecycle step 2.
"""

from __future__ import annotations

import re

from .core.config import RETRIEVE_TOP_K
from .core.context import Context
from .core.kb import KB
from .core.schema import Claim, Evidence, RawEvent

_TOKEN_RE = re.compile(r"[A-Za-z]+\d*|\d+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _score(claim: Claim, query: set[str]) -> int:
    """Lexical overlap between the query and a claim's text + tags. A trivial but
    deterministic stand-in for embeddings (peptide/target tokens dominate)."""
    hay = _tokens(claim.text) | {t.lower() for t in claim.ontology_tags}
    return len(query & hay)


def retrieve(kb: KB, event: RawEvent, evidence: Evidence, k: int | None = None) -> Context:
    """Select the top-k relevant claims + neighborhood and pack a Context."""
    k = k or RETRIEVE_TOP_K
    query = _tokens(event.raw_text)
    for key in ("assay", "metric", "units", "organism", "cell_line"):
        v = evidence.fields.get(key)
        if isinstance(v, str):
            query |= _tokens(v)

    ranked = sorted(
        kb.claims.values(),
        key=lambda c: (_score(c, query), c.id),
        reverse=True,
    )
    top = [c for c in ranked if _score(c, query) > 0][:k]
    if not top:
        top = ranked[:k]  # nothing matched — hand over the k most recent by id

    top_ids = {c.id for c in top}
    edges = [
        e
        for e in kb.live_edges()
        if e.src in top_ids or e.dst in top_ids
    ]
    # pull in 1-hop neighbor claims referenced by those edges
    neighbor_ids = {e.src for e in edges} | {e.dst for e in edges}
    claims = list(top)
    for cid in sorted(neighbor_ids - top_ids):
        c = kb.get_claim(cid)
        if c is not None:
            claims.append(c)

    src = kb.get_source(evidence.source_id)
    sources = [src] if src is not None else []

    return Context(
        event=event,
        evidence=evidence,
        claims=claims,
        sources=sources,
        edges=edges,
    )
