from __future__ import annotations

from decimal import Decimal

import pytest

from lang_token_bench.cli import main
from lang_token_bench.counters.base import CounterRequestError, CounterUnavailableError
from lang_token_bench.openrouter_credits import (
    OpenRouterCredits,
    OpenRouterCreditsClient,
    parse_credits_response,
)


def test_openrouter_credits_parse_remaining_credits() -> None:
    credits = parse_credits_response(
        {
            "total_credits": 10,
            "total_usage": "2.75",
        }
    )

    assert credits.total_credits == Decimal("10")
    assert credits.total_usage == Decimal("2.75")
    assert credits.remaining_credits == Decimal("7.25")


def test_openrouter_credits_parse_nested_data_response() -> None:
    credits = parse_credits_response(
        {
            "data": {
                "total_credits": "5.50",
                "total_usage": "1.25",
            }
        }
    )

    assert credits.remaining_credits == Decimal("4.25")


def test_openrouter_credits_missing_api_key_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(CounterUnavailableError, match="OPENROUTER_API_KEY"):
        OpenRouterCreditsClient().fetch()


def test_openrouter_credits_invalid_response_shape_is_actionable() -> None:
    with pytest.raises(CounterRequestError, match="numeric total_credits and total_usage"):
        parse_credits_response({"total_credits": 10})


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "unauthorized"),
        (403, "forbidden"),
        (429, "rate-limited"),
        (500, "currently unavailable"),
        (503, "currently unavailable"),
    ],
)
def test_openrouter_credits_http_errors_are_actionable(
    status_code: int,
    expected: str,
) -> None:
    class FakeResponse:
        def __init__(self, response_status_code: int) -> None:
            self.status_code = response_status_code

        def json(self):
            return {"error": {"message": "test error detail"}}

    with pytest.raises(CounterRequestError, match=expected):
        OpenRouterCreditsClient()._raise_for_status(FakeResponse(status_code), "test-key")


def test_openrouter_credits_cli_prints_mocked_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_fetch(self):
        return OpenRouterCredits(
            total_credits=Decimal("10.5"),
            total_usage=Decimal("2.25"),
        )

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)

    exit_code = main(["openrouter", "credits"], load_env=False)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        "OpenRouter credits:\n"
        "Total credits: 10.5\n"
        "Total usage: 2.25\n"
        "Remaining credits: 8.25\n"
    )
    assert captured.err == ""


def test_openrouter_credits_cli_missing_key_does_not_print_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    exit_code = main(["openrouter", "credits"], load_env=False)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "OPENROUTER_API_KEY is not set" in captured.err
    assert captured.out == ""


def test_openrouter_credits_cli_redacts_api_key_from_error_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "test-secret-key-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    class FakeResponse:
        status_code = 400

        def json(self):
            return {"error": {"message": f"bad key {secret}"}}

    def fake_get_json(self, *, headers, api_key):
        self._raise_for_status(FakeResponse(), api_key)
        raise AssertionError("unreachable")

    monkeypatch.setattr(OpenRouterCreditsClient, "_get_json", fake_get_json)

    exit_code = main(["openrouter", "credits"], load_env=False)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert secret not in captured.out
    assert secret not in captured.err
    assert "[redacted]" in captured.err
