from __future__ import annotations

from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


class HuggingFaceTokenizerCounter(TokenCounter):
    name = "hf-tokenizer"
    counting_method = "hf_tokenizer"

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        raise NotImplementedError(
            "hf_tokenizer counter is not implemented yet. "
            "It is an optional future reference backend and is not required "
            "for the current OpenRouter observed usage workflow."
        )
