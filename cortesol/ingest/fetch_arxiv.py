"""Fetch real abstracts across MANY scientific fields from arXiv.

This is the multi-field companion to :mod:`cortesol.ingest.fetch_papers` (which
covers biomed via PubMed). A generalist critical-appraisal model must read papers
from EVERY field, so the corpus deliberately spans physics, CS, math, economics,
statistics and quantitative biology. Each abstract is tagged with its `field` — a
label used ONLY for the held-out-field eval split, never shown to the model.

Epistemic note (same as fetch_papers): a fetched abstract is a *Source report* —
EVIDENCE, not truth. Every record carries no oracle. The `field` tag exists so the
splitter can hold entire fields out of training; the appraisal itself is judged on
METHOD (design/statistics/flags), never on subject, so the field carries no label
signal (see `train/appraisal_dataset.py`).

Network + cache: hits the public arXiv Atom API (no key needed) and appends new
records to `data/corpus/arxiv_papers.jsonl`, deduped by arXiv id. Re-runs are
incremental; the cached JSONL is the deterministic input every downstream builder
reads, so nothing offline ever depends on the network.

Usage:
    uv run python -m cortesol.ingest.fetch_arxiv --per-field 40
    uv run python -m cortesol.ingest.fetch_arxiv --fields physics,cs --per-field 20
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
POLITE_DELAY_S = 3.1  # arXiv asks for <= 1 request / 3s
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class ArxivField:
    """One discipline, mapped to the arXiv categories that represent it. The
    `name` is the field tag; `categories` are OR-ed in the search query."""

    name: str
    categories: tuple[str, ...]


# A broad, deliberately diverse spread of disciplines. No field dominates, and the
# set intentionally mixes empirical (q-bio, physics) with theoretical/computational
# (math, cs, stat) so study designs — not subjects — drive the appraisal label.
ARXIV_FIELDS: tuple[ArxivField, ...] = (
    ArxivField("physics", ("cond-mat.supr-con", "hep-ex", "quant-ph")),
    ArxivField("cs", ("cs.LG", "cs.CL", "cs.CV")),
    ArxivField("math", ("math.PR", "math.ST", "math.OC")),
    ArxivField("econ", ("econ.EM", "econ.GN")),
    ArxivField("stat", ("stat.ME", "stat.AP")),
    ArxivField("q-bio", ("q-bio.BM", "q-bio.NC", "q-bio.PE")),
)

FIELDS_BY_NAME: dict[str, ArxivField] = {f.name: f for f in ARXIV_FIELDS}


def _get(params: dict[str, str]) -> bytes:
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
                return resp.read()
        except Exception:  # transient network / rate-limit -> back off
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _arxiv_id(entry: ET.Element) -> str:
    raw = _text(entry.find(f"{_ATOM}id"))  # e.g. http://arxiv.org/abs/2401.00001v2
    tail = raw.rsplit("/", 1)[-1]
    return tail.split("v")[0] if tail else raw


def parse_feed(xml_bytes: bytes, field: str) -> list[dict]:
    """Parse an arXiv Atom feed into paper records tagged with `field`."""
    root = ET.fromstring(xml_bytes)
    papers: list[dict] = []
    for entry in root.findall(f"{_ATOM}entry"):
        abstract = _text(entry.find(f"{_ATOM}summary"))
        title = _text(entry.find(f"{_ATOM}title"))
        if not abstract or not title:
            continue
        authors = [
            _text(a.find(f"{_ATOM}name"))
            for a in entry.findall(f"{_ATOM}author")
            if _text(a.find(f"{_ATOM}name"))
        ]
        published = _text(entry.find(f"{_ATOM}published"))
        papers.append(
            {
                "arxiv_id": _arxiv_id(entry),
                "field": field,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": published[:4] if published else "",
                "venue": "arxiv",
            }
        )
    return papers


def search(field: ArxivField, per_field: int) -> list[dict]:
    """Query arXiv for one field's most recent abstracts (sorted for stability)."""
    query = " OR ".join(f"cat:{c}" for c in field.categories)
    raw = _get(
        {
            "search_query": query,
            "start": "0",
            "max_results": str(per_field),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    papers = parse_feed(raw, field.name)
    papers.sort(key=lambda p: p["arxiv_id"])  # stable within a run
    return papers


def collect(fields: list[ArxivField], per_field: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for field in fields:
        print(f"  {field.name:<9} querying arXiv ({', '.join(field.categories)}) ...", flush=True)
        try:
            papers = search(field, per_field)
        except Exception as exc:  # keep going; one field failing must not abort the run
            print(f"    ! failed for {field.name}: {exc}", flush=True)
            continue
        added = 0
        for paper in papers:
            if paper["arxiv_id"] in seen:
                continue
            seen.add(paper["arxiv_id"])
            out.append(paper)
            added += 1
        print(f"    -> {added} new abstracts", flush=True)
        time.sleep(POLITE_DELAY_S)
    return out


def _load_cached(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows, {str(r.get("arxiv_id")) for r in rows}


def write_outputs(new_rows: list[dict], out_dir: Path) -> dict:
    """Append newly-fetched rows to the cached JSONL (deduped) and rewrite the
    manifest. Returns the manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "arxiv_papers.jsonl"
    existing, seen = _load_cached(path)
    merged = list(existing)
    for row in new_rows:
        if str(row["arxiv_id"]) not in seen:
            seen.add(str(row["arxiv_id"]))
            merged.append(row)
    merged.sort(key=lambda p: (p["field"], p["arxiv_id"]))
    with path.open("w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_field: dict[str, int] = {}
    for row in merged:
        by_field[row["field"]] = by_field.get(row["field"], 0) + 1
    manifest = {
        "total_papers": len(merged),
        "by_field": dict(sorted(by_field.items())),
        "note": "multi-field arXiv abstracts (evidence, no oracle); field tag is eval-only",
    }
    (out_dir / "arxiv_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(merged)} arXiv papers -> {path}")
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-field", type=int, default=40, help="abstracts per field")
    ap.add_argument("--fields", type=str, default="", help="comma-separated subset by name")
    ap.add_argument("--out", type=str, default="data/corpus", help="output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.fields:
        wanted = {f.strip().lower() for f in args.fields.split(",")}
        fields = [FIELDS_BY_NAME[n] for n in wanted if n in FIELDS_BY_NAME]
        if not fields:
            raise SystemExit(
                f"no known arXiv fields in {sorted(wanted)}; known: {sorted(FIELDS_BY_NAME)}"
            )
    else:
        fields = list(ARXIV_FIELDS)
    print(f"Fetching up to {args.per_field} abstracts each for {len(fields)} fields ...\n")
    rows = collect(fields, args.per_field)
    write_outputs(rows, Path(args.out))


if __name__ == "__main__":
    main()
