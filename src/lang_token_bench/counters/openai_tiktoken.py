from __future__ import annotations

from lang_token_bench.counters.base import CounterUnavailableError, TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


TIKTOKEN_INSTALL_MESSAGE = (
    "openai-tiktoken counter requires the optional dependency. "
    "Install it with: uv sync --extra tiktoken"
)


class OpenAITiktokenCounter(TokenCounter):
    name = "openai-tiktoken"
    counting_method = "openai_tiktoken"
    default_encoding = "o200k_base"

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        try:
            import tiktoken
        except ImportError as exc:
            raise CounterUnavailableError(TIKTOKEN_INSTALL_MESSAGE) from exc

        encoding_name = (
            self.default_encoding
            if model is None or model.tokenizer_name is None
            else model.tokenizer_name
        )
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # pragma: no cover - depends on optional package data
            model_id = "<none>" if model is None else model.id
            raise CounterUnavailableError(
                f"Failed to load tiktoken encoding '{encoding_name}' for model '{model_id}': {exc}"
            ) from exc

        return TokenCountResult(
            token_count=len(encoding.encode(text)),
            counter=self.name,
            counting_method=self.counting_method,
            model_id=None if model is None else model.id,
            tokenizer_name=encoding_name,
        )

