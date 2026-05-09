from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any

from lang_token_bench.counters.base import CounterRequestError, CounterUnavailableError
from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://lang-token-bench.local"
OPENROUTER_TITLE = "Language Token Efficiency Benchmark"
OPENROUTER_INSTALL_MESSAGE = (
    "openrouter-usage counter requires the optional dependency. "
    "Install it with: uv sync --extra openrouter"
)
MAX_ERROR_DETAIL_LENGTH = 1000
DEFAULT_MAX_OUTPUT_TOKENS = 16


@dataclass(frozen=True)
class OpenRouterProviderRouting:
    only: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    order: tuple[str, ...] = ()
    allow_fallbacks: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.only:
            payload["only"] = list(self.only)
        if self.ignore:
            payload["ignore"] = list(self.ignore)
        if self.order:
            payload["order"] = list(self.order)
        if self.allow_fallbacks is not None:
            payload["allow_fallbacks"] = self.allow_fallbacks
        return payload

    def is_empty(self) -> bool:
        return not self.to_payload()


class OpenRouterUsageCounter(TokenCounter):
    name = "openrouter-usage"
    counting_method = "openrouter_usage"

    def __init__(
        self,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        provider_routing: OpenRouterProviderRouting | None = None,
    ) -> None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer.")
        self.max_output_tokens = max_output_tokens
        self.provider_routing = provider_routing or OpenRouterProviderRouting()

    def get_api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        resolved_key = self.get_api_key() if api_key is None else api_key.strip()
        if not resolved_key:
            raise CounterUnavailableError(
                "OPENROUTER_API_KEY is not set in the environment. "
                "Set it in the OS environment or local .env before using "
                "the openrouter_usage backend."
            )
        return {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }

    def build_payload(self, text: str, model: ModelConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": 0,
        }
        provider_payload = self.provider_routing.to_payload()
        if provider_payload:
            payload["provider"] = provider_payload
        return payload

    def extract_prompt_tokens(self, response_json: dict[str, Any]) -> int:
        usage = response_json.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("OpenRouter response did not include a usage object.")

        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            raise ValueError("OpenRouter response usage.prompt_tokens was missing or invalid.")
        return prompt_tokens

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        if model is None:
            raise ValueError("openrouter_usage counter requires a ModelConfig with an OpenRouter model id.")

        api_key = self.get_api_key()
        headers = self.build_headers(api_key)
        payload = self.build_payload(text, model)
        response_json = self._post_json(payload=payload, headers=headers, api_key=api_key)
        prompt_tokens = self.extract_prompt_tokens(response_json)

        return TokenCountResult(
            token_count=prompt_tokens,
            counter=self.name,
            counting_method=self.counting_method,
            model_id=model.id,
            tokenizer_name=model.tokenizer_name,
        )

    def _post_json(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        api_key: str = "",
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise CounterUnavailableError(OPENROUTER_INSTALL_MESSAGE) from exc

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise CounterRequestError(
                _redact_secret(
                    _format_transport_error(
                        "OpenRouter Chat Completions request timed out",
                        payload,
                    ),
                    api_key,
                )
            ) from exc
        except httpx.RequestError as exc:
            raise CounterRequestError(
                _redact_secret(
                    _format_transport_error(
                        f"OpenRouter Chat Completions request failed: {type(exc).__name__}: {exc}",
                        payload,
                    ),
                    api_key,
                )
            ) from exc

        if response.status_code >= 400:
            raise CounterRequestError(
                _redact_secret(_format_http_error(response, payload), api_key)
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise CounterRequestError("OpenRouter response was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise CounterRequestError("OpenRouter response JSON was not an object.")
        return data

    def _extract_error_detail(self, response: Any) -> str:
        return _extract_error_detail(response)


def _format_http_error(response: Any, payload: dict[str, Any]) -> str:
    status_code = int(getattr(response, "status_code", 0))
    model_id = str(payload.get("model", "unknown"))
    prefix = _http_status_prefix(status_code)
    detail = _extract_error_detail(response)
    return (
        f"{prefix} Endpoint: {OPENROUTER_API_URL}. "
        f"Model: {model_id}. Details: {detail}"
    )


def _http_status_prefix(status_code: int) -> str:
    if status_code == 400:
        return "OpenRouter Chat Completions request was rejected (HTTP 400)."
    if status_code == 401:
        return (
            "OpenRouter Chat Completions request was unauthorized (HTTP 401). "
            "Check OPENROUTER_API_KEY."
        )
    if status_code == 403:
        return (
            "OpenRouter Chat Completions request was forbidden (HTTP 403). "
            "Check account access, model access, and OPENROUTER_API_KEY."
        )
    if status_code == 429:
        return (
            "OpenRouter Chat Completions request was rate-limited (HTTP 429). "
            "Try again later or reduce request volume."
        )
    if 500 <= status_code <= 599:
        return (
            f"OpenRouter Chat Completions API is currently unavailable (HTTP {status_code}). "
            "Try again later."
        )
    return f"OpenRouter Chat Completions API returned HTTP {status_code}."


def _format_transport_error(message: str, payload: dict[str, Any]) -> str:
    model_id = str(payload.get("model", "unknown"))
    return f"{message}. Endpoint: {OPENROUTER_API_URL}. Model: {model_id}."


def _extract_error_detail(response: Any) -> str:
    try:
        data = response.json()
    except ValueError:
        text = str(getattr(response, "text", "")).strip()
        return _truncate(text or "No error detail returned.")

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            parts = []
            for key in ("message", "code", "type", "metadata"):
                if key in error:
                    parts.append(f"error.{key}: {_format_safe_value(error[key])}")
            if parts:
                return _truncate("; ".join(parts))
        if error is not None:
            return _truncate(f"error: {_format_safe_value(error)}")
        if "message" in data:
            return _truncate(f"message: {_format_safe_value(data['message'])}")
    return _truncate(_format_safe_value(data))


def _format_safe_value(value: Any) -> str:
    value = _sanitize_value(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                sanitized[str(key)] = "[redacted]"
            else:
                sanitized[str(key)] = _sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    sensitive_fragments = (
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "secret",
        "password",
        "request_headers",
    )
    return normalized == "headers" or any(fragment in normalized for fragment in sensitive_fragments)


def _truncate(value: str) -> str:
    if len(value) <= MAX_ERROR_DETAIL_LENGTH:
        return value
    return value[:MAX_ERROR_DETAIL_LENGTH].rstrip() + "... [truncated]"


def _redact_secret(message: str, secret: str) -> str:
    if not secret:
        return message
    return message.replace(secret, "[redacted]")
