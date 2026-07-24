"""Shared LLM-provider resolution for Cortesol's generative seams (area C).

Cortesol's belief engine never calls an LLM — ``core/`` is deterministic and
offline. The *consumer* features that do use a model — the Research chat agent,
grounded report generation, import-a-field drafting (all in ``ui/``), and the
training teacher (``train/``) — may run on any of three interchangeable
providers. This module is the ONE place that decides which:

    anthropic  — Claude, via the official ``anthropic`` SDK (the default)
    openai     — via the ``openai`` SDK
    gemini     — via the ``openai`` SDK pointed at Google's OpenAI-compatible URL

Precedence is **Claude → OpenAI → Gemini** (the first provider whose API key is
configured), overridable with ``LLM_PROVIDER``. With no key for any provider the
feature degrades gracefully to its deterministic/offline fallback, exactly as
before — the belief graph never depends on a model being reachable.

Nothing here imports an SDK: the seam that needs a client constructs it lazily,
so importing this module (or the whole package) never requires ``anthropic`` or
``openai`` to be installed. Keys are read through the shared ``.env`` loader in
``ingest/extract`` so a local ``.env`` and the real environment behave the same.
"""

from __future__ import annotations

from .ingest.extract import _env  # shared .env loader / key reader

ANTHROPIC = "anthropic"
OPENAI = "openai"
GEMINI = "gemini"

# Precedence order: Claude first, then OpenAI, then Gemini.
ORDER: tuple[str, ...] = (ANTHROPIC, OPENAI, GEMINI)

_KEY_ENV = {
    ANTHROPIC: "ANTHROPIC_API_KEY",
    OPENAI: "OPENAI_API_KEY",
    GEMINI: "GEMINI_API_KEY",
}

# Default base URLs for the two OpenAI-SDK providers (Anthropic uses its own SDK
# default). Each is overridable via the matching *_BASE_URL env var.
_OPENAI_COMPAT_BASE = {
    OPENAI: "https://api.openai.com/v1",
    GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai/",
}
_BASE_ENV = {OPENAI: "OPENAI_BASE_URL", GEMINI: "GEMINI_BASE_URL"}


def provider_key(provider: str) -> str | None:
    """The configured API key for ``provider`` (env or ``.env``), or None."""
    env_name = _KEY_ENV.get(provider)
    return _env(env_name) if env_name else None


def active_provider() -> str | None:
    """The provider to use this call, or None when no key is configured anywhere.

    Honours an explicit ``LLM_PROVIDER`` override only when that provider also has
    a key; otherwise falls back to the first key present in :data:`ORDER`
    (Claude → OpenAI → Gemini). Read fresh each call so tests / hot-reloads that
    change the environment take effect immediately.
    """
    override = (_env("LLM_PROVIDER") or "").strip().lower()
    if override in ORDER and provider_key(override):
        return override
    for provider in ORDER:
        if provider_key(provider):
            return provider
    return None


def any_available() -> bool:
    """True when at least one generative provider is configured."""
    return active_provider() is not None


def is_anthropic(provider: str | None) -> bool:
    """Whether ``provider`` is Claude (uses the native ``anthropic`` SDK path)."""
    return provider == ANTHROPIC


def openai_base_url(provider: str) -> str:
    """Base URL for an OpenAI-SDK provider (``openai`` / ``gemini``).

    Never called for Anthropic — that path uses the ``anthropic`` SDK's own host.
    """
    default = _OPENAI_COMPAT_BASE[provider]
    return _env(_BASE_ENV[provider], default) or default
