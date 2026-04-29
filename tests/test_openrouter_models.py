from __future__ import annotations

import pytest

from lang_token_bench.cli import main
from lang_token_bench.counters.base import CounterRequestError, CounterUnavailableError
from lang_token_bench.openrouter_models import (
    OpenRouterModelsClient,
    parse_openrouter_model_ids,
)


def _write_suite_config(path, model_ids: list[str]) -> None:
    lines = [
        "suites:",
        "  - name: test_suite",
        "    description: Test suite",
        "    model_ids:",
    ]
    lines.extend(f"      - {model_id}" for model_id in model_ids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_openrouter_model_ids_parse_response() -> None:
    model_ids = parse_openrouter_model_ids(
        {
            "data": [
                {"id": "openai/gpt-4o-mini"},
                {"id": "openai/gpt-4o"},
            ]
        }
    )

    assert model_ids == {"openai/gpt-4o-mini", "openai/gpt-4o"}


def test_openrouter_model_ids_invalid_shape_is_actionable() -> None:
    with pytest.raises(CounterRequestError, match="data list"):
        parse_openrouter_model_ids({"data": {}})


def test_openrouter_validate_models_cli_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openai/gpt-4o-mini", "openai/gpt-4o"])

    def fake_fetch_model_ids(self):
        return {"openai/gpt-4o-mini", "openai/gpt-4o"}

    monkeypatch.setattr(OpenRouterModelsClient, "fetch_model_ids", fake_fetch_model_ids)

    exit_code = main(
        [
            "openrouter",
            "validate-models",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OpenRouter model validation for suite: test_suite" in captured.out
    assert "All suite model IDs were found in OpenRouter." in captured.out
    assert captured.err == ""


def test_openrouter_validate_models_cli_reports_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openai/gpt-4o-mini", "openai/gpt-4o"])

    def fake_fetch_model_ids(self):
        return {"openai/gpt-4o-mini"}

    monkeypatch.setattr(OpenRouterModelsClient, "fetch_model_ids", fake_fetch_model_ids)

    exit_code = main(
        [
            "openrouter",
            "validate-models",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Missing model IDs:" in captured.out
    assert "openai/gpt-4o" in captured.out
    assert "were not found in OpenRouter" in captured.err


def test_openrouter_models_missing_api_key_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(CounterUnavailableError, match="OPENROUTER_API_KEY"):
        OpenRouterModelsClient().fetch_model_ids()
