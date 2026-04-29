from __future__ import annotations

import os
from typing import Any

from lang_token_bench.counters.base import CounterRequestError, CounterUnavailableError
from lang_token_bench.openrouter_credits import (
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
)


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODELS_INSTALL_MESSAGE = (
    "OpenRouter model validation requires the optional dependency. "
    "Install it with: uv sync --extra openrouter"
)


class OpenRouterModelsClient:
    def get_api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        resolved_key = self.get_api_key() if api_key is None else api_key.strip()
        if not resolved_key:
            raise CounterUnavailableError(
                "OPENROUTER_API_KEY is not set in the environment. "
                "Set it in the OS environment or local .env before running "
                "`lang-token-bench openrouter validate-models`."
            )
        return {
            "Authorization": f"Bearer {resolved_key}",
            "Accept": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }

    def fetch_model_ids(self) -> set[str]:
        api_key = self.get_api_key()
        headers = self.build_headers(api_key)
        response_json = self._get_json(headers=headers, api_key=api_key)
        return parse_openrouter_model_ids(response_json)

    def _get_json(self, *, headers: dict[str, str], api_key: str) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise CounterUnavailableError(OPENROUTER_MODELS_INSTALL_MESSAGE) from exc

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(OPENROUTER_MODELS_URL, headers=headers)
        except httpx.TimeoutException as exc:
            raise CounterRequestError("OpenRouter models request timed out.") from exc
        except httpx.RequestError as exc:
            message = f"OpenRouter models request failed: {type(exc).__name__}: {exc}"
            raise CounterRequestError(_redact_secret(message, api_key)) from exc

        self._raise_for_status(response, api_key)

        try:
            data = response.json()
        except ValueError as exc:
            raise CounterRequestError("OpenRouter models response was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise CounterRequestError("OpenRouter models response JSON was not an object.")
        return data

    def _raise_for_status(self, response: Any, api_key: str) -> None:
        status_code = int(getattr(response, "status_code", 0))
        if status_code < 400:
            return

        if status_code == 401:
            raise CounterRequestError(
                "OpenRouter models request was unauthorized (HTTP 401). "
                "Check OPENROUTER_API_KEY."
            )
        if status_code == 403:
            raise CounterRequestError(
                "OpenRouter models request was forbidden (HTTP 403). "
                "Check account access and OPENROUTER_API_KEY."
            )
        if status_code == 429:
            raise CounterRequestError(
                "OpenRouter models request was rate-limited (HTTP 429). Try again later."
            )
        if 500 <= status_code <= 599:
            raise CounterRequestError(
                f"OpenRouter models API is currently unavailable (HTTP {status_code}). "
                "Try again later."
            )

        detail = _redact_secret(_extract_error_detail(response), api_key)
        raise CounterRequestError(f"OpenRouter models API returned HTTP {status_code}: {detail}")


def parse_openrouter_model_ids(response_json: dict[str, Any]) -> set[str]:
    raw_data = response_json.get("data")
    if not isinstance(raw_data, list):
        raise CounterRequestError("OpenRouter models response did not include a data list.")

    model_ids: set[str] = set()
    for item in raw_data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise CounterRequestError("OpenRouter models response included an invalid model entry.")
        model_ids.add(item["id"])
    return model_ids


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
