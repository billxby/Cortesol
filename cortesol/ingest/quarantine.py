"""Quarantine + spotlighting — untrusted text handling (area A, deterministic).

Reference: Research/"Prompt Injection Defense" §Layer-1.
Datamark/delimit incoming text, strip anything resembling system delimiters, and
guarantee it only ever occupies a DATA position downstream — never an instruction.
Input is always RawEvent.untrusted_view() (gold already stripped, PD6).
"""

from __future__ import annotations

from ..core.schema import Evidence, RawEvent


def quarantine(event: RawEvent) -> Evidence:
    """Wrap/clean the untrusted event and parse it into a structured Evidence."""
    ...  # TODO
