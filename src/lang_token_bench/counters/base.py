from __future__ import annotations

from abc import ABC, abstractmethod

from lang_token_bench.schema import ModelConfig, TokenCountResult


class CounterUnavailableError(RuntimeError):
    """Raised when an optional counter dependency is not installed."""


class CounterRequestError(RuntimeError):
    """Raised when an API-backed counter request fails."""


class TokenCounter(ABC):
    name: str
    counting_method: str

    @abstractmethod
    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        """Count tokens for text and return normalized counter metadata."""
