"""CORTEX format adapter (area C) — the ONLY file that knows CORTEX's KB/stream
format. Write it FIRST at kickoff once the real format is known (Project Plan H0).

Translates CORTEX's belief-KB serialization -> our KB, and their result stream ->
our RawEvent (with sim_meta = None, since real events carry no gold). If their
format differs from assumptions, ONLY this file changes.
"""

from __future__ import annotations

from ..core.kb import KB
from ..core.schema import RawEvent


def load_kb(path: str) -> KB:
    """Parse CORTEX's belief KB into our KB."""
    ...  # TODO at kickoff


def load_stream(path: str) -> list[RawEvent]:
    """Parse CORTEX's result stream into RawEvents."""
    ...  # TODO at kickoff
