"""Shared test collection config.

`test_training_pipeline.py` imports `freesolo` at module top. That package lives in
the optional `train` extra (`uv sync --extra train`), so the default dev env
(`make install` / `make test` / the `test-contract` merge gate) does not have it.
Without this guard, that one unconditional import raises at COLLECTION time, which
aborts the entire pytest run — including the contract gate — before any test runs.

When `freesolo` is absent we drop that module from collection; when the `train`
extra is installed it collects and runs normally.
"""

from __future__ import annotations

import importlib.util

collect_ignore: list[str] = []
if importlib.util.find_spec("freesolo") is None:
    collect_ignore.append("unit/test_training_pipeline.py")
