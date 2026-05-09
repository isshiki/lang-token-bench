from __future__ import annotations

from lang_token_bench.counters.anthropic_api import AnthropicApiCounter
from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.counters.gemini_api import GeminiApiCounter
from lang_token_bench.counters.hf_tokenizer import HuggingFaceTokenizerCounter
from lang_token_bench.counters.openai_tiktoken import OpenAITiktokenCounter
from lang_token_bench.counters.openrouter_usage import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    OpenRouterProviderRouting,
    OpenRouterUsageCounter,
)
from lang_token_bench.counters.playwright_web import PlaywrightWebCounter
from lang_token_bench.counters.simple_counter import SimpleCounter


def create_counter(
    name: str,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    openrouter_provider_routing: OpenRouterProviderRouting | None = None,
) -> TokenCounter:
    normalized = name.strip().lower()
    factories: dict[str, type[TokenCounter]] = {
        "simple": SimpleCounter,
        "openai-tiktoken": OpenAITiktokenCounter,
        "anthropic-api": AnthropicApiCounter,
        "anthropic_api": AnthropicApiCounter,
        "gemini-api": GeminiApiCounter,
        "gemini_api": GeminiApiCounter,
        "hf-tokenizer": HuggingFaceTokenizerCounter,
        "hf_tokenizer": HuggingFaceTokenizerCounter,
        "playwright-web": PlaywrightWebCounter,
        "playwright_web": PlaywrightWebCounter,
    }
    if normalized in {"openrouter-usage", "openrouter_usage"}:
        return OpenRouterUsageCounter(
            max_output_tokens=max_output_tokens,
            provider_routing=openrouter_provider_routing,
        )
    try:
        return factories[normalized]()
    except KeyError as exc:
        available = ", ".join(sorted([*factories, "openrouter-usage", "openrouter_usage"]))
        raise ValueError(f"Unknown counter '{name}'. Available counters: {available}") from exc


__all__ = ["TokenCounter", "create_counter"]
