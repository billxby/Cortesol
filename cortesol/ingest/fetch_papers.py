"""Fetch real peptide-paper abstracts from PubMed and store them as events.

This is a *data-acquisition* utility (Area B/C), NOT part of the deterministic
core. It hits the network, so it lives in `ingest/`, never in `core/`.

What it does:
  1. For each of a curated list of leading research/therapeutic peptides, query
     PubMed (NCBI E-utilities, no API key needed) for recent abstracts.
  2. Parse title / abstract / journal / year / authors / doi out of the XML.
  3. Write two artifacts under `data/papers/`:
       - `raw_papers.jsonl`     one full paper record per line (all metadata)
       - `peptide_events.jsonl` RawEvent-shaped events for the pipeline
       - `manifest.json`        counts + per-source tiers, for a quick overview

Epistemic note (docs/ONTOLOGY.md §Truth vs Belief): a fetched paper is a
*Source report* — EVIDENCE, not truth. Every emitted event has `sim_meta: null`
(no gold, no oracle). A trusted journal gets a high source tier so its evidence
moves belief more, but the engine still earns confidence through the capped math;
nothing here stamps a claim as "true". Real-paper streams therefore support the
live demo, not the scored calibration metrics (those need the simulator's oracle).

Usage:
    uv run python -m cortesol.ingest.fetch_papers --per-peptide 8 --email you@x.com
    uv run python -m cortesol.ingest.fetch_papers --peptides semaglutide,bpc-157
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# --- The curated peptide list -------------------------------------------------
# "Leading peptides people are using" — therapeutic + research peptides with a
# known primary target, so downstream ops-extraction has a binding claim to hang
# on. `query_terms` narrow PubMed toward the peptide's own literature.


@dataclass(frozen=True)
class Peptide:
    name: str  # canonical name used in tags / ids
    aliases: tuple[str, ...]  # other names / brand names searched in PubMed
    target: str  # primary molecular target (for a seed binding claim)
    indication: str  # what it is studied/used for


LEADING_PEPTIDES: tuple[Peptide, ...] = (
    Peptide("semaglutide", ("Ozempic", "Wegovy"), "GLP1R", "type 2 diabetes / obesity"),
    Peptide("tirzepatide", ("Mounjaro", "Zepbound"), "GLP1R/GIPR", "type 2 diabetes / obesity"),
    Peptide("retatrutide", ("LY3437943",), "GLP1R/GIPR/GCGR", "type 2 diabetes / obesity"),
    Peptide("liraglutide", ("Victoza", "Saxenda"), "GLP1R", "type 2 diabetes / obesity"),
    Peptide("exenatide", ("Byetta", "exendin-4"), "GLP1R", "type 2 diabetes"),
    Peptide("cagrilintide", (), "amylin receptor", "obesity"),
    Peptide("setmelanotide", ("Imcivree",), "MC4R", "genetic obesity"),
    Peptide("bremelanotide", ("Vyleesi", "PT-141"), "MC4R", "sexual dysfunction"),
    Peptide("melanotan-II", ("melanotan 2", "MT-II"), "MC1R/MC4R", "tanning / research"),
    Peptide("teriparatide", ("Forteo", "PTH 1-34"), "PTH1R", "osteoporosis"),
    Peptide("octreotide", ("Sandostatin",), "SSTR2", "acromegaly / neuroendocrine tumors"),
    Peptide("linaclotide", ("Linzess",), "guanylate cyclase-C", "IBS-C / constipation"),
    Peptide("ziconotide", ("Prialt",), "N-type calcium channel", "chronic pain"),
    Peptide("icatibant", ("Firazyr",), "bradykinin B2 receptor", "hereditary angioedema"),
    Peptide("degarelix", ("Firmagon",), "GnRH receptor", "prostate cancer"),
    Peptide("enfuvirtide", ("Fuzeon", "T-20"), "gp41", "HIV fusion inhibition"),
    Peptide("elamipretide", ("SS-31", "MTP-131"), "cardiolipin", "mitochondrial disease"),
    Peptide("bpc-157", ("body protection compound 157",), "unknown", "tissue repair (research)"),
    Peptide("thymosin-beta-4", ("TB-500", "thymosin b4"), "actin", "tissue repair (research)"),
    Peptide("ghk-cu", ("copper tripeptide-1", "GHK copper"), "unknown", "skin / wound healing"),
    Peptide("selank", (), "unknown", "anxiolytic (research)"),
)

# --- Source-tier heuristic ----------------------------------------------------
# Real papers are peer-reviewed => `reputable` by default; a small prestige set
# gets `top_journal`; preprint servers get `preprint`. Tiers key SOURCE_PRIORS in
# core/config.py, which sets each source's belief-move cap log(tau/phi).

PRESTIGE_JOURNALS = {
    "nature", "science", "cell", "the lancet", "lancet", "new england journal of medicine",
    "nature medicine", "nature biotechnology", "cell metabolism", "nature communications",
    "pnas", "proceedings of the national academy of sciences", "jama", "nature chemical biology",
}
PREPRINT_MARKERS = ("biorxiv", "medrxiv", "preprint", "research square", "ssrn", "chemrxiv")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "cortesol-peptide-fetch"
POLITE_DELAY_S = 0.34  # <= 3 req/s without an API key (NCBI policy)


def source_tier(journal: str) -> str:
    j = journal.strip().lower()
    if any(m in j for m in PREPRINT_MARKERS):
        return "preprint"
    if j in PRESTIGE_JOURNALS:
        return "top_journal"
    return "reputable"


# --- E-utilities client (stdlib only) ----------------------------------------


def _get(path: str, params: dict[str, str], email: str) -> bytes:
    params = {**params, "tool": TOOL_NAME, "email": email}
    url = f"{EUTILS}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # transient network / rate-limit -> back off
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
            _ = exc
    raise RuntimeError("unreachable")


def esearch(term: str, retmax: int, email: str) -> list[str]:
    """Return PMIDs for a query, most-relevant first, abstract required."""
    raw = _get(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": f"({term}) AND hasabstract[text] AND English[lang]",
            "retmax": str(retmax),
            "retmode": "json",
            "sort": "relevance",
        },
        email,
    )
    data = json.loads(raw)
    return data.get("esearchresult", {}).get("idlist", [])


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def efetch(pmids: list[str], email: str) -> list[dict]:
    """Fetch and parse full records for a batch of PMIDs."""
    if not pmids:
        return []
    raw = _get(
        "efetch.fcgi",
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        email,
    )
    root = ET.fromstring(raw)
    papers: list[dict] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//PMID"))
        title = _text(art.find(".//ArticleTitle"))
        # Abstract may be split into labelled sections.
        chunks = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            body = _text(ab)
            chunks.append(f"{label}: {body}" if label else body)
        abstract = " ".join(c for c in chunks if c).strip()
        journal = _text(art.find(".//Journal/Title"))
        year = _text(art.find(".//JournalIssue/PubDate/Year")) or _text(
            art.find(".//JournalIssue/PubDate/MedlineDate")
        )
        authors = []
        for a in art.findall(".//AuthorList/Author"):
            last, fore = _text(a.find("LastName")), _text(a.find("ForeName"))
            if last:
                authors.append(f"{fore} {last}".strip())
        doi = ""
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)
                break
        if not abstract:  # skip records that came back without usable text
            continue
        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
                "doi": doi,
            }
        )
    return papers


# --- Event shaping ------------------------------------------------------------


def to_event(paper: dict, peptide: Peptide, t: int) -> dict:
    """Shape a paper into a RawEvent-shaped dict. `sim_meta` is null: a real paper
    is evidence with no oracle. Structured `fields` carry provenance + hints; the
    prose stays in `raw_text` (quarantined) for the extractor to read."""
    journal = paper["journal"]
    tier = source_tier(journal)
    source_id = f"pubmed:{journal}".strip().lower().replace(" ", "_")
    return {
        "id": f"pmid_{paper['pmid']}",
        "t": t,
        "source_id": source_id,
        "raw_text": f"{paper['title']} {paper['abstract']}".strip(),
        "fields": {
            "pmid": paper["pmid"],
            "doi": paper["doi"],
            "title": paper["title"],
            "journal": journal,
            "year": paper["year"],
            "source_tier": tier,
            "peptide": peptide.name,
            "peptide_aliases": list(peptide.aliases),
            "primary_target": peptide.target,
            "indication": peptide.indication,
        },
        "sim_meta": None,
    }


# --- Orchestration ------------------------------------------------------------


@dataclass
class Result:
    papers: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    seen_pmids: set[str] = field(default_factory=set)


def collect(peptides: list[Peptide], per_peptide: int, email: str) -> Result:
    res = Result()
    t = 0
    for pep in peptides:
        names = " OR ".join(f'"{n}"' for n in (pep.name, *pep.aliases))
        term = f"({names})[Title/Abstract]"
        print(f"  {pep.name:<18} querying PubMed ...", flush=True)
        try:
            pmids = esearch(term, per_peptide, email)
            time.sleep(POLITE_DELAY_S)
            papers = efetch(pmids, email)
            time.sleep(POLITE_DELAY_S)
        except Exception as exc:
            print(f"    ! failed for {pep.name}: {exc}", flush=True)
            continue
        added = 0
        for paper in papers:
            if paper["pmid"] in res.seen_pmids:
                continue
            res.seen_pmids.add(paper["pmid"])
            paper_row = {**paper, "peptide": pep.name, "primary_target": pep.target}
            res.papers.append(paper_row)
            res.events.append(to_event(paper, pep, t))
            t += 1
            added += 1
        print(f"    -> {added} new abstracts", flush=True)
    return res


def write_outputs(res: Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_papers.jsonl"
    ev_path = out_dir / "peptide_events.jsonl"
    manifest_path = out_dir / "manifest.json"

    with raw_path.open("w") as f:
        for row in res.papers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with ev_path.open("w") as f:
        for ev in res.events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    by_tier: dict[str, int] = {}
    by_peptide: dict[str, int] = {}
    for ev in res.events:
        by_tier[ev["fields"]["source_tier"]] = by_tier.get(ev["fields"]["source_tier"], 0) + 1
        by_peptide[ev["fields"]["peptide"]] = by_peptide.get(ev["fields"]["peptide"], 0) + 1
    manifest = {
        "total_events": len(res.events),
        "unique_pmids": len(res.seen_pmids),
        "by_source_tier": by_tier,
        "by_peptide": by_peptide,
        "note": "real-paper evidence; sim_meta is null (no ground-truth oracle)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(res.events)} events -> {ev_path}")
    print(f"Wrote {len(res.papers)} papers -> {raw_path}")
    print(f"Manifest -> {manifest_path}\n{json.dumps(manifest, indent=2)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-peptide", type=int, default=8, help="abstracts per peptide")
    ap.add_argument("--peptides", type=str, default="", help="comma-separated subset by name")
    ap.add_argument(
        "--email",
        type=str,
        default="cortesol@example.com",
        help="contact email for NCBI E-utilities (be polite)",
    )
    ap.add_argument("--out", type=str, default="data/papers", help="output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.peptides:
        wanted = {p.strip().lower() for p in args.peptides.split(",")}
        peptides = [p for p in LEADING_PEPTIDES if p.name.lower() in wanted]
        if not peptides:
            raise SystemExit(f"no known peptides in {sorted(wanted)}")
    else:
        peptides = list(LEADING_PEPTIDES)
    print(f"Fetching up to {args.per_peptide} abstracts each for {len(peptides)} peptides ...\n")
    res = collect(peptides, args.per_peptide, args.email)
    write_outputs(res, Path(args.out))


if __name__ == "__main__":
    main()
