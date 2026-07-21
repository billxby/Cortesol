"""Domain registry + active-domain accessor.

The belief-core reads the ACTIVE domain (default: peptides) through
`get_active_domain()`. Switching fields is `set_active_domain(name_or_domain)` —
the UI rebuilds the KB around it, exactly like switching the data source. The
active pointer is a module global rather than per-KB to keep every engine
signature stable; one process serves one field at a time, which matches the demo
and the training loop.
"""

from __future__ import annotations

from .base import Domain, default_correlation_group
from .peptides import PEPTIDES

_REGISTRY: dict[str, Domain] = {}


def register_domain(domain: Domain) -> Domain:
    """Add (or replace) a domain in the registry, keyed by `domain.name`."""
    _REGISTRY[domain.name] = domain
    return domain


def get_domain(name: str) -> Domain:
    return _REGISTRY[name]


def has_domain(name: str) -> bool:
    return name in _REGISTRY


def list_domains() -> list[Domain]:
    return list(_REGISTRY.values())


# Peptides is registered and active on import, so all existing behavior is
# byte-identical until something calls set_active_domain().
register_domain(PEPTIDES)
_ACTIVE: list[Domain] = [PEPTIDES]


def get_active_domain() -> Domain:
    """The field of knowledge the belief-core currently represents."""
    return _ACTIVE[0]


def set_active_domain(domain: Domain | str) -> Domain:
    """Point the belief-core at a different field. Accepts a `Domain` or a
    registered name; an unregistered `Domain` is registered on the way in.
    Returns the now-active domain."""
    resolved = domain if isinstance(domain, Domain) else _REGISTRY[domain]
    if resolved.name not in _REGISTRY:
        register_domain(resolved)
    _ACTIVE[0] = resolved
    return resolved


__all__ = [
    "Domain",
    "default_correlation_group",
    "PEPTIDES",
    "register_domain",
    "get_domain",
    "has_domain",
    "list_domains",
    "get_active_domain",
    "set_active_domain",
]
