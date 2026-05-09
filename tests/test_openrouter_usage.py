from __future__ import annotations

import pytest

from lang_token_bench.counters.base import CounterUnavailableError
from lang_token_bench.counters.openrouter_usage import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_ERROR_DETAIL_LENGTH,
    OpenRouterProviderRouting,
    OpenRouterUsageCounter,
    _format_http_error,
    _redact_secret,
)
from lang_token_bench.schema import ModelConfig


def test_openrouter_payload_uses_observed_usage_method_shape() -> None:
    counter = OpenRouterUsageCounter()
    model = ModelConfig(
        id="openai/gpt-4o",
        provider="openrouter",
        display_name="GPT-4o via OpenRouter",
        counter="openrouter-usage",
        tokenizer_name=None,
        input_price_per_1m_tokens=None,
        enabled=False,
    )

    payload = counter.build_payload("Hello", model)

    assert payload["model"] == "openai/gpt-4o"
    assert payload["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert "provider" not in payload


def test_openrouter_payload_accepts_custom_max_output_tokens() -> None:
    counter = OpenRouterUsageCounter(max_output_tokens=32)
    model = ModelConfig(
        id="openai/gpt-4o",
        provider="openrouter",
        display_name="GPT-4o via OpenRouter",
        counter="openrouter-usage",
        tokenizer_name=None,
        input_price_per_1m_tokens=None,
        enabled=False,
    )

    payload = counter.build_payload("Hello", model)

    assert payload["max_tokens"] == 32


def test_openrouter_payload_accepts_provider_routing() -> None:
    counter = OpenRouterUsageCounter(
        provider_routing=OpenRouterProviderRouting(
            only=("anthropic",),
            order=("anthropic", "amazon-bedrock"),
            allow_fallbacks=False,
        )
    )
    model = ModelConfig(
        id="anthropic/claude-opus-4.7",
        provider="openrouter",
        display_name="Claude Opus 4.7 via OpenRouter",
        counter="openrouter-usage",
        tokenizer_name=None,
        input_price_per_1m_tokens=None,
        enabled=False,
    )

    payload = counter.build_payload("Hello", model)

    assert payload["provider"] == {
        "only": ["anthropic"],
        "order": ["anthropic", "amazon-bedrock"],
        "allow_fallbacks": False,
    }


def test_openrouter_extract_prompt_tokens() -> None:
    counter = OpenRouterUsageCounter()

    assert counter.extract_prompt_tokens({"usage": {"prompt_tokens": 42}}) == 42


def test_openrouter_headers_require_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = OpenRouterUsageCounter()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(CounterUnavailableError, match="OPENROUTER_API_KEY"):
        counter.build_headers()


def test_openrouter_count_parses_mocked_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = OpenRouterUsageCounter()
    model = ModelConfig(
        id="openai/gpt-4o-mini",
        provider="openrouter",
        display_name="GPT-4o mini via OpenRouter",
        counter="openrouter-usage",
        tokenizer_name=None,
        input_price_per_1m_tokens=None,
        enabled=False,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-printed")

    def fake_post_json(*, payload, headers, api_key=""):
        assert payload["model"] == "openai/gpt-4o-mini"
        assert payload["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
        assert headers["Authorization"].startswith("Bearer ")
        assert api_key == "test-key-not-printed"
        return {"usage": {"prompt_tokens": 17}}

    monkeypatch.setattr(counter, "_post_json", fake_post_json)

    result = counter.count("Hello", model)

    assert result.token_count == 17
    assert result.counter == "openrouter-usage"
    assert result.counting_method == "openrouter_usage"


def test_openrouter_400_json_error_includes_safe_fields_and_redacts_secret() -> None:
    secret = "test-secret-key-value"

    class FakeResponse:
        status_code = 400

        def json(self):
            return {
                "error": {
                    "message": f"Provider returned error for {secret}",
                    "code": "provider_error",
                    "type": "invalid_request_error",
                    "metadata": {
                        "provider": "example",
                        "reason": "model unavailable",
                        "headers": {"Authorization": f"Bearer {secret}"},
                    },
                }
            }

    message = _redact_secret(
        _format_http_error(FakeResponse(), {"model": "openrouter/model-a"}),
        secret,
    )

    assert "HTTP 400" in message
    assert "Endpoint: https://openrouter.ai/api/v1/chat/completions" in message
    assert "Model: openrouter/model-a" in message
    assert "error.message: Provider returned error" in message
    assert "error.code: provider_error" in message
    assert "error.type: invalid_request_error" in message
    assert "error.metadata:" in message
    assert '"headers": "[redacted]"' in message
    assert secret not in message
    assert "[redacted]" in message


def test_openrouter_non_json_error_body_is_truncated() -> None:
    class FakeResponse:
        status_code = 502
        text = "x" * (MAX_ERROR_DETAIL_LENGTH + 100)

        def json(self):
            raise ValueError("not JSON")

    message = _format_http_error(FakeResponse(), {"model": "openrouter/model-a"})

    assert "HTTP 502" in message
    assert "Model: openrouter/model-a" in message
    assert "[truncated]" in message
    assert len(message) < MAX_ERROR_DETAIL_LENGTH + 300
