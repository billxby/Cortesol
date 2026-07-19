"""The Research chat agent (area C) — a Gemini-Flash assistant with two tools.

This is a *consumer* of the belief graph, not a new write path. The chat model
never holds or sets belief: it orchestrates two tools and grounds its prose in what
the deterministic engine returns.

  1. `find_papers`         — reuse `ingest/fetch_papers` (PubMed, no API key).
  2. `add_to_belief_graph` — add found papers to the library and run each through the
                             REAL lifecycle (`pipeline.process_event`) so the engine
                             assesses truthfulness; return the earned confidences.

The model PROPOSES which papers to fetch and assess; the ledger DISPOSES the belief.
Untrusted paper text enters only as `raw_text` data via the normal quarantine path.

Gemini is reached through the already-installed `openai` SDK pointed at Google's
OpenAI-compatible endpoint (`GEMINI_BASE_URL`), mirroring the Flash seam in
`ingest/extract.py`. Streaming + `tools=[...]` function-calling are standard.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from ..adapters import cortex
from ..core.domain import peptide_facts
from ..ingest import fetch_papers
from ..ingest.extract import _env

# --------------------------------------------------------------------------
# Gemini client (OpenAI-compatible endpoint) — lazily constructed singleton.
# --------------------------------------------------------------------------

_client = None
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_MODEL = "gemini-flash-latest"
MAX_ROUNDS = 6  # cap tool-calling rounds so a loop can never run away


def gemini_available() -> bool:
    """True when a Gemini API key is configured (in .env or the environment)."""
    return bool(_env("GEMINI_API_KEY"))


def _model() -> str:
    return _env("GEMINI_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL


def _email() -> str:
    """Contact email for NCBI E-utilities (be polite). Reuse the Crossref one."""
    return _env("CROSSREF_MAILTO") or "cortesol@example.com"


def _get_client():
    """Lazy AsyncOpenAI client aimed at Gemini's OpenAI-compatible endpoint.
    Returns None when no key is configured, so the route degrades gracefully."""
    global _client
    if _client is None:
        key = _env("GEMINI_API_KEY")
        if not key:
            return None
        from openai import AsyncOpenAI  # lazy: importing this module never needs the SDK

        _client = AsyncOpenAI(
            base_url=_env("GEMINI_BASE_URL", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL,
            api_key=key,
        )
    return _client


# --------------------------------------------------------------------------
# System prompt — STATIC and trusted. The user's messages and paper text are the
# only untrusted content; they never authorize an action here either.
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are Cortesol Research, a peptide-research assistant with an epistemic "
    "immune system. You do NOT trust the raw text of papers. Your source of truth "
    "is the Cortesol belief graph — a deterministic engine that assigns each claim a "
    "calibrated confidence after screening for fraud, hype, weak study designs, and "
    "source reliability. A published claim is EVIDENCE, never truth.\n\n"
    "For any substantive question about a peptide, follow this workflow:\n"
    "1. Identify the peptide and its primary molecular target.\n"
    "2. Call find_papers(query, peptide, target) to retrieve the literature.\n"
    "3. Call add_to_belief_graph(pmids) to add those papers to the library and run "
    "each through the belief engine. It returns, per claim, the engine-earned "
    "confidence (0-1), evidence counts (r supporting / s contradicting), status, and "
    "any red flags.\n"
    "4. Answer using ONLY those returned confidences. State the confidence "
    "explicitly (e.g. 'confidence 0.94'). Explicitly flag anything that is "
    "low-confidence, uncertain, flagged out-of-distribution, refused, or screened as "
    "hype/fraud. Never assert a paper's claim is true just because it was "
    "published.\n\n"
    "Be concise, precise, and honest about uncertainty. Cite papers by title/journal "
    "when relevant. If the belief graph holds no confident claim, say so plainly "
    "rather than speculating. For pure chit-chat you may answer directly, but any "
    "factual claim about a peptide must be grounded via the tools."
)

# --------------------------------------------------------------------------
# Tool schemas (OpenAI `tools=[...]` shape). These are the ONLY actions the model
# can name — a closed vocabulary, just like the op schema for the extractor.
# --------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_papers",
            "description": (
                "Search PubMed for peer-reviewed abstracts about a peptide. Returns "
                "lightweight paper references (pmid, title, journal, year, source "
                "tier). Call this before add_to_belief_graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text PubMed query, e.g. 'semaglutide GLP1R binding' "
                            "or 'BPC-157 tendon healing'."
                        ),
                    },
                    "peptide": {
                        "type": "string",
                        "description": "Canonical peptide name, e.g. 'semaglutide' or 'bpc-157'.",
                    },
                    "target": {
                        "type": "string",
                        "description": (
                            "Primary molecular target of the peptide, e.g. 'GLP1R'. "
                            "Use 'none' if the peptide has no single well-defined receptor "
                            "target (its efficacy claims carry it instead)."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many papers to fetch (1-12).",
                    },
                },
                "required": ["query", "peptide", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_belief_graph",
            "description": (
                "Add previously found papers to the Cortesol library and run each "
                "through the belief engine to assess truthfulness. Returns, per claim, "
                "the engine-earned confidence (0-1), evidence counts, status, and red "
                "flags. Answer the user using these confidences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pmids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "PMIDs to add (from find_papers). Omit or leave empty to "
                            "add every paper found so far this turn."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]

# add_papers callback: given enriched paper dicts, mutate the KB (under lock) and
# return {"claims": [...readout for the model...], "papers": [...for the client...]}.
AddPapers = Callable[[list[dict]], Awaitable[dict]]


# --------------------------------------------------------------------------
# Tool bodies
# --------------------------------------------------------------------------


def _paper_ref(p: dict) -> dict:
    """Lightweight reference handed back to the model (no abstract — it stays
    server-side so untrusted prose never round-trips through the model)."""
    return {
        "pmid": p.get("pmid"),
        "title": p.get("title"),
        "journal": p.get("journal"),
        "year": p.get("year"),
        "tier": cortex.journal_to_tier(p.get("journal")),
    }


def _resolve_target(peptide: str, target: str) -> str:
    """The grounded primary target for a peptide: the model's value if real, else the
    curated one (core/domain.PEPTIDE_KNOWLEDGE, then LEADING_PEPTIDES). Returns "" —
    never "unknown" — when the peptide has no single confirmed receptor; the KB then
    seeds its efficacy claims rather than a meaningless "binds unknown" node."""
    if target and target.strip().lower() not in ("", "unknown", "none", "n/a", "null"):
        return target.strip()
    pep = (peptide or "").strip().lower()
    facts = peptide_facts(pep)
    if facts is not None:
        return facts.get("target") or ""
    for p in fetch_papers.LEADING_PEPTIDES:
        names = (p.name.lower(), *(a.lower() for a in p.aliases))
        if pep and pep in names:
            return p.target or ""
    return ""


def _find_papers(args: dict) -> list[dict]:
    """Blocking PubMed fetch (run under asyncio.to_thread). Attaches the peptide +
    primary_target the model supplied, which the KB seeding needs downstream."""
    query = str(args.get("query") or "").strip()
    peptide = str(args.get("peptide") or "").strip()
    target = _resolve_target(peptide, str(args.get("target") or ""))
    try:
        k = int(args.get("k") or 6)
    except (TypeError, ValueError):
        k = 6
    k = max(1, min(k, 12))

    term = query or peptide
    if not term:
        return []
    pmids = fetch_papers.esearch(term, k, _email())
    papers = fetch_papers.efetch(pmids, _email())
    return [{**p, "peptide": peptide or None, "primary_target": target} for p in papers]


# --------------------------------------------------------------------------
# The agentic loop — streaming completions with tool-calling.
# --------------------------------------------------------------------------


def _sanitize_history(messages: list[dict] | None) -> list[dict]:
    """Keep only well-formed user/assistant text turns from the client, capped."""
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out[-20:]


def _parse_args(raw: str) -> dict:
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


async def run_chat(
    messages: list[dict],
    *,
    add_papers: AddPapers,
) -> AsyncIterator[dict]:
    """Drive one chat turn. Yields typed event dicts for the SSE stream:
      {type:"token", text}         — a streamed chunk of the assistant's answer
      {type:"tool_start", name, args}
      {type:"tool_end", name, summary, count?}
      {type:"papers", items}       — papers added + their engine verdicts (for cards)
      {type:"error", text}
      {type:"done"}
    Blocking work (PubMed, KB) runs off the event loop; KB writes hold STATE.lock
    inside the `add_papers` callback provided by the caller.
    """
    client = _get_client()
    if client is None:
        yield {
            "type": "error",
            "text": "Gemini is not configured — set GEMINI_API_KEY in .env, then reload.",
        }
        return

    convo: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    convo += _sanitize_history(messages)
    if not any(m["role"] == "user" for m in convo):
        yield {"type": "error", "text": "No question to answer."}
        return

    model = _model()
    pending: dict[str, dict] = {}  # pmid -> enriched paper dict, filled by find_papers

    try:
        for _round in range(MAX_ROUNDS):
            content_parts: list[str] = []
            tool_calls: dict[int, dict] = {}  # index -> {id, name, args}
            stream = await client.chat.completions.create(
                model=model,
                messages=convo,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                    yield {"type": "token", "text": delta.content}
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = tool_calls.setdefault(
                        tc.index, {"id": None, "name": "", "args": "", "extra": None}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn and fn.name:
                        slot["name"] = fn.name
                    if fn and fn.arguments:
                        slot["args"] += fn.arguments
                    # Gemini attaches a `thought_signature` (under extra_content) to each
                    # function call; it MUST be echoed back with the tool-call turn or the
                    # next request 400s. Capture it from whichever chunk carries it.
                    extra = getattr(tc, "model_extra", None) or {}
                    if extra.get("extra_content"):
                        slot["extra"] = extra["extra_content"]

            if not tool_calls:
                yield {"type": "done"}
                return

            # Replay the assistant's tool-call turn back into the conversation.
            convo.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": tc["id"] or f"call_{i}",
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["args"] or "{}"},
                            # echo Gemini's thought_signature back on the tool call
                            **({"extra_content": tc["extra"]} if tc.get("extra") else {}),
                        }
                        for i, tc in sorted(tool_calls.items())
                    ],
                }
            )

            for i, tc in sorted(tool_calls.items()):
                name = tc["name"]
                args = _parse_args(tc["args"])
                call_id = tc["id"] or f"call_{i}"
                yield {"type": "tool_start", "name": name, "args": args}

                if name == "find_papers":
                    found = await asyncio.to_thread(_find_papers, args)
                    for p in found:
                        if p.get("pmid"):
                            pending[str(p["pmid"])] = p
                    model_result = {"papers": [_paper_ref(p) for p in found]}
                    yield {
                        "type": "tool_end",
                        "name": name,
                        "summary": f"Found {len(found)} paper(s) on PubMed",
                        "count": len(found),
                    }
                elif name == "add_to_belief_graph":
                    want = [str(m) for m in (args.get("pmids") or [])]
                    selected = (
                        [pending[m] for m in want if m in pending]
                        if want
                        else list(pending.values())
                    )
                    if not selected:
                        model_result = {"error": "no pending papers — call find_papers first"}
                        yield {"type": "tool_end", "name": name, "summary": "No papers to add"}
                    else:
                        readout = await add_papers(selected)
                        # The model grounds its answer in these engine-earned numbers.
                        model_result = {
                            "claims": readout.get("claims", []),
                            "papers": [
                                {
                                    "title": p.get("title"),
                                    "journal": p.get("journal"),
                                    "verdict": p.get("verdict"),
                                    "red_flags": p.get("red_flags", []),
                                }
                                for p in readout.get("papers", [])
                            ],
                        }
                        if readout.get("papers"):
                            yield {"type": "papers", "items": readout["papers"]}
                        if readout.get("claims"):
                            yield {"type": "assessment", "items": readout["claims"]}
                        yield {
                            "type": "tool_end",
                            "name": name,
                            "summary": f"Assessed {len(selected)} paper(s) in the belief graph",
                            "count": len(selected),
                        }
                else:
                    model_result = {"error": f"unknown tool {name}"}
                    yield {"type": "tool_end", "name": name, "summary": "unknown tool"}

                convo.append(
                    {"role": "tool", "tool_call_id": call_id, "content": json.dumps(model_result)}
                )

        # Hit the round cap — force one final, tool-free answer so the turn always ends
        # with a written response rather than dangling on tool calls.
        stream = await client.chat.completions.create(
            model=model, messages=convo, temperature=0.3, stream=True
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield {"type": "token", "text": delta.content}
        yield {"type": "done"}
    except Exception as exc:  # network / API / decode — surface once, never crash the route
        yield {"type": "error", "text": f"Model error: {type(exc).__name__}: {exc}"}
        return
