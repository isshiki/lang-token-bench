from __future__ import annotations

from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


class AnthropicApiCounter(TokenCounter):
    name = "anthropic-api"
    counting_method = "anthropic_token_count_api"

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        raise NotImplementedError(
            "anthropic_api counter is not implemented yet. "
            "It is an optional future reference backend and is not required "
            "for the current OpenRouter observed usage workflow."
        )
