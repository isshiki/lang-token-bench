from __future__ import annotations

from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


class GeminiApiCounter(TokenCounter):
    name = "gemini-api"
    counting_method = "gemini_count_tokens_api"

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        raise NotImplementedError(
            "gemini_api counter is not implemented yet. "
            "It is an optional future reference backend and is not required "
            "for the current OpenRouter observed usage workflow."
        )
