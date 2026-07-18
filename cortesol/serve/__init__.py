"""Serving adapters for local and remote Cortesol proposal models."""

from .ollama import OllamaExtractor, OllamaProposalClient

__all__ = ["OllamaExtractor", "OllamaProposalClient"]
