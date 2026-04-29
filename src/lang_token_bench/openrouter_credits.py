from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from lang_token_bench.counters.base import CounterRequestError, CounterUnavailableError


OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
OPENROUTER_REFERER = "https://lang-token-bench.local"
OPENROUTER_TITLE = "Language Token Efficiency Benchmark"
OPENROUTER_CREDITS_INSTALL_MESSAGE = (
    "OpenRouter credit checks require the optional dependency. "
    "Install it with: uv sync --extra openrouter"
)
OPENROUTER_CREDITS_SHAPE_ERROR = (
    "OpenRouter credits response did not include numeric total_credits and total_usage."
)


@dataclass(frozen=True)
class OpenRouterCredits:
    total_credits: Decimal
    total_usage: Decimal

    @property
    def remaining_credits(self) -> Decimal:
        return self.total_credits - self.total_usage


class OpenRouterCreditsClient:
    def get_api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        resolved_key = self.get_api_key() if api_key is None else api_key.strip()
        if not resolved_key:
            raise CounterUnavailableError(
                "OPENROUTER_API_KEY is not set in the environment. "
                "Set it in the OS environment or local .env before running "
                "`lang-token-bench openrouter credits`."
            )
        return {
            "Authorization": f"Bearer {resolved_key}",
            "Accept": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }

    def fetch(self) -> OpenRouterCredits:
        api_key = self.get_api_key()
        headers = self.build_headers(api_key)
        response_json = self._get_json(headers=headers, api_key=api_key)
        return parse_credits_response(response_json)

    def _get_json(self, *, headers: dict[str, str], api_key: str) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise CounterUnavailableError(OPENROUTER_CREDITS_INSTALL_MESSAGE) from exc

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(OPENROUTER_CREDITS_URL, headers=headers)
        except httpx.TimeoutException as exc:
            raise CounterRequestError("OpenRouter credits request timed out.") from exc
        except httpx.RequestError as exc:
            message = f"OpenRouter credits request failed: {type(exc).__name__}: {exc}"
            raise CounterRequestError(_redact_secret(message, api_key)) from exc

        self._raise_for_status(response, api_key)

        try:
            data = response.json()
        except ValueError as exc:
            raise CounterRequestError("OpenRouter credits response was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise CounterRequestError("OpenRouter credits response JSON was not an object.")
        return data

    def _raise_for_status(self, response: Any, api_key: str) -> None:
        status_code = int(getattr(response, "status_code", 0))
        if status_code < 400:
            return

        if status_code == 401:
            raise CounterRequestError(
                "OpenRouter credits request was unauthorized (HTTP 401). "
                "Check OPENROUTER_API_KEY."
            )
        if status_code == 403:
            raise CounterRequestError(
                "OpenRouter credits request was forbidden (HTTP 403). "
                "Check account access and OPENROUTER_API_KEY."
            )
        if status_code == 429:
            raise CounterRequestError(
                "OpenRouter credits request was rate-limited (HTTP 429). "
                "Try again later."
            )
        if 500 <= status_code <= 599:
            raise CounterRequestError(
                f"OpenRouter credits API is currently unavailable (HTTP {status_code}). "
                "Try again later."
            )

        detail = _redact_secret(_extract_error_detail(response), api_key)
        raise CounterRequestError(
            f"OpenRouter credits API returned HTTP {status_code}: {detail}"
        )


def parse_credits_response(response_json: dict[str, Any]) -> OpenRouterCredits:
    raw_data = response_json.get("data")
    data = raw_data if isinstance(raw_data, dict) else response_json

    return OpenRouterCredits(
        total_credits=_read_decimal_field(data, "total_credits"),
        total_usage=_read_decimal_field(data, "total_usage"),
    )


def format_credit_amount(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _read_decimal_field(data: dict[str, Any], key: str) -> Decimal:
    value = data.get(key)
    if isinstance(value, bool) or value is None:
        raise CounterRequestError(OPENROUTER_CREDITS_SHAPE_ERROR)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CounterRequestError(OPENROUTER_CREDITS_SHAPE_ERROR) from exc


def _extract_error_detail(response: Any) -> str:
    try:
        data = response.json()
    except ValueError:
        text = getattr(response, "text", "")
        return str(text).strip()[:500] or "No error detail returned."

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)[:500]
        if error:
            return str(error)[:500]
        message = data.get("message")
        if message:
            return str(message)[:500]
    return str(data)[:500]


def _redact_secret(message: str, secret: str) -> str:
    if not secret:
        return message
    return message.replace(secret, "[redacted]")
