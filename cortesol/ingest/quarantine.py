"""Quarantine + spotlighting — untrusted text handling (area A, deterministic).

Reference: Research/"Prompt Injection Defense" §Layer-1.
Datamark/delimit incoming text, strip anything resembling system delimiters, and
guarantee it only ever occupies a DATA position downstream — never an instruction.
Input is always RawEvent.untrusted_view() (gold already stripped, PD6).
"""

from __future__ import annotations

from ..core.domain import correlation_group
from ..core.schema import Evidence, RawEvent


def quarantine(event: RawEvent) -> Evidence:
    """Wrap/clean the untrusted event and parse it into a structured Evidence."""
    if event.sim_meta is not None:
        raise ValueError("quarantine accepts only RawEvent.untrusted_view()")
    text = event.raw_text.replace("\x00", "").replace("</UNTRUSTED_DATA>", "[end marker removed]")
    marked = f"<UNTRUSTED_DATA>\n{text}\n</UNTRUSTED_DATA>"
    fields = dict(event.fields)
    return Evidence(
        id=event.id,
        source_id=event.source_id,
        raw_text=marked,
        fields=fields,
        correlation_group=correlation_group(fields),
    )
