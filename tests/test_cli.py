from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

from lang_token_bench.cli import main
from lang_token_bench.counters.base import CounterRequestError
from lang_token_bench.counters.openrouter_usage import OpenRouterUsageCounter
from lang_token_bench.openrouter_credits import OpenRouterCredits, OpenRouterCreditsClient
from lang_token_bench.reporters.csv_reporter import write_csv_report
from lang_token_bench.schema import BenchmarkResult, TokenCountResult


def _write_suite_config(path: Path, model_ids: list[str]) -> None:
    lines = [
        "suites:",
        "  - name: test_suite",
        "    description: Test suite",
        "    model_ids:",
    ]
    lines.extend(f"      - {model_id}" for model_id in model_ids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_saved_run_result(
    output_dir: Path,
    *,
    model_id: str,
    text_id: str = "short_instruction",
    language_code: str = "en",
) -> None:
    write_csv_report(
        [
            BenchmarkResult(
                model_id=model_id,
                provider="openrouter",
                counter="openrouter-usage",
                counting_method="openrouter_usage",
                language_code=language_code,
                language_name="English",
                text_id=text_id,
                token_count=10,
                ratio_to_english=1.0,
                input_price_per_1m_tokens=None,
                estimated_input_cost_usd=None,
                timestamp_utc="2026-04-29T00:00:00Z",
            )
        ],
        output_dir / "runs" / "existing" / "results.csv",
    )


def test_openrouter_dry_run_does_not_call_api(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_called(self, text, model=None):
        raise AssertionError("OpenRouter count should not be called during dry-run")

    def fail_credits_if_called(self):
        raise AssertionError("OpenRouter credits should not be called during dry-run")

    monkeypatch.setattr(OpenRouterUsageCounter, "count", fail_if_called)
    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_credits_if_called)

    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--dry-run",
            "--model-id",
            "openai/gpt-4o-mini",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dry run: no token counting or API requests were performed." in captured.out
    assert "Planned benchmark rows: 1" in captured.out
    assert "Max output tokens: 16" in captured.out
    assert "openai/gpt-4o-mini" in captured.out
    assert captured.err == ""


def test_simple_run_does_not_call_openrouter_credits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fail_credits_if_called(self):
        raise AssertionError("OpenRouter credits should not be called for simple counter")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_credits_if_called)

    exit_code = main(
        [
            "run",
            "--counter",
            "simple",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    assert not (tmp_path / "run_summary.json").exists()
    assert not (tmp_path / "run_history.csv").exists()
    assert not (tmp_path / "runs").exists()


def test_openrouter_requires_yes_without_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "openai/gpt-4o-mini",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert (
        "openrouter-usage requires --yes to run real API requests. "
        "Use --dry-run to preview."
    ) in captured.err


def test_openrouter_run_tracks_credits_and_writes_run_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "test-secret-key-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    credit_calls: list[str] = []

    def fake_fetch(self):
        credit_calls.append("fetch")
        if len(credit_calls) == 1:
            return OpenRouterCredits(
                total_credits=Decimal("10"),
                total_usage=Decimal("2"),
            )
        return OpenRouterCredits(
            total_credits=Decimal("10"),
            total_usage=Decimal("2.25"),
        )

    count_calls: list[str] = []

    def fake_count(self, text, model=None):
        count_calls.append(text)
        return TokenCountResult(
            token_count=3,
            counter="openrouter-usage",
            counting_method="openrouter_usage",
            model_id=model.id if model else None,
            tokenizer_name=None,
        )

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fake_count)

    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "openai/gpt-4o-mini",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert credit_calls == ["fetch", "fetch"]
    assert len(count_calls) == 1
    assert "OpenRouter credits before:" in captured.out
    assert "OpenRouter credit summary:" in captured.out
    assert "Credits before: 8" in captured.out
    assert "Credits after: 7.75" in captured.out
    assert "Credits used: 0.25" in captured.out
    assert secret not in captured.out
    assert secret not in captured.err

    summary_path = tmp_path / "run_summary.json"
    history_path = tmp_path / "run_history.csv"
    assert summary_path.exists()
    assert history_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["counter"] == "openrouter-usage"
    assert summary["model_id"] == "openai/gpt-4o-mini"
    assert summary["text_id"] == "short_instruction"
    assert summary["rows_executed"] == 1
    assert summary["credits_before_remaining"] == "8"
    assert summary["credits_after_remaining"] == "7.75"
    assert summary["credits_used"] == "0.25"

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert "openai-gpt-4o-mini" in run_dir.name
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "results.md").exists()
    assert (run_dir / "run_summary.json").exists()
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary == summary

    with history_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["run_id"] == summary["run_id"]
    assert rows[0]["credits_used"] == "0.25"


def test_openrouter_run_max_output_tokens_option_updates_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key-value")
    payloads: list[dict] = []

    def fake_fetch(self):
        return OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2"))

    def fake_post_json(self, *, payload, headers, api_key=""):
        payloads.append(payload)
        return {"usage": {"prompt_tokens": 17}}

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "_post_json", fake_post_json)

    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "openai/gpt-4o-mini",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--max-output-tokens",
            "32",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    assert [payload["max_tokens"] for payload in payloads] == [32]


def test_openrouter_run_provider_routing_options_update_payload_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "test-secret-key-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    payloads: list[dict] = []

    def fake_fetch(self):
        return OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2"))

    def fake_post_json(self, *, payload, headers, api_key=""):
        payloads.append(payload)
        return {"usage": {"prompt_tokens": 17}}

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "_post_json", fake_post_json)

    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "anthropic/claude-opus-4.7",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--provider-only",
            "anthropic",
            "--no-provider-fallbacks",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert [payload["provider"] for payload in payloads] == [
        {"only": ["anthropic"], "allow_fallbacks": False}
    ]
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["provider_routing"] == {
        "only": ["anthropic"],
        "allow_fallbacks": False,
    }
    assert secret not in captured.out
    assert secret not in captured.err


def test_openrouter_provider_routing_dry_run_is_displayed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_credits_if_called(self):
        raise AssertionError("Credits API should not be called during dry-run")

    def fail_count_if_called(self, text, model=None):
        raise AssertionError("Usage API should not be called during dry-run")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_credits_if_called)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fail_count_if_called)

    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "anthropic/claude-opus-4.7",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--provider-ignore",
            "amazon-bedrock",
            "--dry-run",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OpenRouter provider routing:" in captured.out
    assert "- ignore: amazon-bedrock" in captured.out


def test_openrouter_provider_only_and_ignore_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "anthropic/claude-opus-4.7",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--provider-only",
            "anthropic",
            "--provider-ignore",
            "amazon-bedrock",
            "--dry-run",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--provider-only and --provider-ignore cannot be used together." in captured.err


def test_openrouter_credit_failure_stops_usage_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_fetch(self):
        raise CounterRequestError("OpenRouter credits request timed out.")

    def fail_count_if_called(self, text, model=None):
        raise AssertionError("OpenRouter usage should not run when credits check fails")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fail_count_if_called)

    exit_code = main(
        [
            "run",
            "--counter",
            "openrouter-usage",
            "--model-id",
            "openai/gpt-4o-mini",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "OpenRouter credits request timed out." in captured.err
    assert not (tmp_path / "run_summary.json").exists()
    assert not (tmp_path / "run_history.csv").exists()


def test_run_suite_dry_run_does_not_call_openrouter_apis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openrouter/model-a", "openrouter/model-b"])

    def fail_credits_if_called(self):
        raise AssertionError("Credits API should not be called during run-suite dry-run")

    def fail_count_if_called(self, text, model=None):
        raise AssertionError("Usage API should not be called during run-suite dry-run")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_credits_if_called)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fail_count_if_called)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--dry-run",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Suite dry run" in captured.out
    assert "Planned benchmark rows: 2" in captured.out
    assert "Force: false" in captured.out
    assert "Max output tokens: 16" in captured.out
    assert "Models to run:" in captured.out
    assert "Models to skip:" in captured.out
    assert "- openrouter/model-a: 1" in captured.out
    assert "- openrouter/model-b: 1" in captured.out
    assert captured.err == ""


def test_run_suite_requires_yes_for_openrouter_models(tmp_path) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openrouter/model-a"])

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
        ],
        load_env=False,
    )

    assert exit_code == 2


def test_run_suite_executes_models_in_order_and_writes_suite_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    model_ids = ["openrouter/model-a", "openrouter/model-b"]
    _write_suite_config(suite_path, model_ids)
    secret = "test-secret-key-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    credit_values = iter(
        [
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2")),
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2.1")),
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2.1")),
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2.3")),
        ]
    )

    def fake_fetch(self):
        return next(credit_values)

    count_order: list[str] = []

    def fake_count(self, text, model=None):
        count_order.append(model.id)
        return TokenCountResult(
            token_count=3,
            counter="openrouter-usage",
            counting_method="openrouter_usage",
            model_id=model.id if model else None,
            tokenizer_name=None,
        )

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fake_count)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert count_order == model_ids
    assert secret not in captured.out
    assert secret not in captured.err
    assert "Suite summary:" in captured.out

    suite_dirs = list((tmp_path / "suite_runs").iterdir())
    assert len(suite_dirs) == 1
    suite_summary_path = suite_dirs[0] / "suite_summary.json"
    suite_summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    assert suite_summary["suite_name"] == "test_suite"
    assert suite_summary["models_requested"] == model_ids
    assert suite_summary["models_completed"] == model_ids
    assert suite_summary["models_skipped"] == []
    assert suite_summary["models_failed"] == []
    assert suite_summary["failure_reasons"] == {}
    assert suite_summary["total_rows_executed"] == 2
    assert suite_summary["credits_before_remaining"] == "8"
    assert suite_summary["credits_after_remaining"] == "7.7"
    assert suite_summary["credits_used"] == "0.3"
    assert len(suite_summary["run_ids"]) == 2

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2
    with (tmp_path / "run_history.csv").open("r", encoding="utf-8", newline="") as file:
        history_rows = list(csv.DictReader(file))
    assert len(history_rows) == 2


def test_run_suite_max_output_tokens_option_updates_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openrouter/model-a"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key-value")
    payloads: list[dict] = []

    def fake_fetch(self):
        return OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2"))

    def fake_post_json(self, *, payload, headers, api_key=""):
        payloads.append(payload)
        return {"usage": {"prompt_tokens": 17}}

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "_post_json", fake_post_json)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--max-output-tokens",
            "32",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    assert [payload["max_tokens"] for payload in payloads] == [32]


def test_run_suite_provider_routing_option_updates_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openrouter/model-a"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key-value")
    payloads: list[dict] = []

    def fake_fetch(self):
        return OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2"))

    def fake_post_json(self, *, payload, headers, api_key=""):
        payloads.append(payload)
        return {"usage": {"prompt_tokens": 17}}

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "_post_json", fake_post_json)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--provider-order",
            "anthropic,amazon-bedrock",
            "--no-provider-fallbacks",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    assert [payload["provider"] for payload in payloads] == [
        {"order": ["anthropic", "amazon-bedrock"], "allow_fallbacks": False}
    ]


def test_run_suite_model_id_filter_limits_suite_models(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openrouter/model-a", "openrouter/model-b"])

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--model-id",
            "openrouter/model-b",
            "--dry-run",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "openrouter/model-a" not in captured.out
    assert "openrouter/model-b" in captured.out


def test_run_suite_language_code_filter_limits_languages(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path, ["openrouter/model-a"])

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--language-code",
            "en,hi",
            "--dry-run",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Planned benchmark rows: 2" in captured.out
    assert "- en (English)" in captured.out
    assert "- hi (Hindi)" in captured.out
    assert "- ja (Japanese)" not in captured.out


def test_run_suite_skips_complete_saved_results_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    model_id = "openrouter/model-a"
    _write_suite_config(suite_path, [model_id])
    _write_saved_run_result(tmp_path, model_id=model_id)

    def fail_credits_if_called(self):
        raise AssertionError("Credits API should not be called for skipped model")

    def fail_count_if_called(self, text, model=None):
        raise AssertionError("Usage API should not be called for skipped model")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_credits_if_called)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fail_count_if_called)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Skipping suite models with complete saved run results:" in captured.out
    assert f"- {model_id}" in captured.out
    assert "No suite models need to run" in captured.out

    suite_dirs = list((tmp_path / "suite_runs").iterdir())
    assert len(suite_dirs) == 1
    suite_summary = json.loads(
        (suite_dirs[0] / "suite_summary.json").read_text(encoding="utf-8")
    )
    assert suite_summary["models_requested"] == [model_id]
    assert suite_summary["models_completed"] == []
    assert suite_summary["models_skipped"] == [model_id]
    assert suite_summary["models_failed"] == []
    assert suite_summary["failure_reasons"] == {}
    assert suite_summary["total_rows_executed"] == 0
    assert suite_summary["run_ids"] == []


def test_run_suite_force_runs_even_with_complete_saved_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    model_id = "openrouter/model-a"
    _write_suite_config(suite_path, [model_id])
    _write_saved_run_result(tmp_path, model_id=model_id)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key-value")

    credit_values = iter(
        [
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2")),
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2.1")),
        ]
    )

    def fake_fetch(self):
        return next(credit_values)

    count_calls: list[str] = []

    def fake_count(self, text, model=None):
        count_calls.append(model.id)
        return TokenCountResult(
            token_count=3,
            counter="openrouter-usage",
            counting_method="openrouter_usage",
            model_id=model.id if model else None,
            tokenizer_name=None,
        )

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fake_count)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--force",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Force mode" in captured.out
    assert count_calls == [model_id]

    suite_dirs = list((tmp_path / "suite_runs").iterdir())
    assert len(suite_dirs) == 1
    suite_summary = json.loads(
        (suite_dirs[0] / "suite_summary.json").read_text(encoding="utf-8")
    )
    assert suite_summary["models_completed"] == [model_id]
    assert suite_summary["models_skipped"] == []
    assert suite_summary["failure_reasons"] == {}
    assert suite_summary["total_rows_executed"] == 1


def test_run_suite_dry_run_shows_skip_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    skipped_model_id = "openrouter/model-a"
    planned_model_id = "openrouter/model-b"
    _write_suite_config(suite_path, [skipped_model_id, planned_model_id])
    _write_saved_run_result(tmp_path, model_id=skipped_model_id)

    def fail_credits_if_called(self):
        raise AssertionError("Credits API should not be called during run-suite dry-run")

    def fail_count_if_called(self, text, model=None):
        raise AssertionError("Usage API should not be called during run-suite dry-run")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fail_credits_if_called)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fail_count_if_called)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--dry-run",
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Planned benchmark rows: 1" in captured.out
    assert "Models to run:" in captured.out
    assert f"- {planned_model_id}" in captured.out
    assert "Models to skip:" in captured.out
    assert f"- {skipped_model_id}" in captured.out


def test_run_suite_failure_reports_model_id_and_writes_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    model_id = "openrouter/model-a"
    _write_suite_config(suite_path, [model_id])
    secret = "test-secret-key-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    def fake_fetch(self):
        return OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2"))

    def fake_count(self, text, model=None):
        raise CounterRequestError("OpenRouter Chat Completions request was rejected.")

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fake_count)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert f"Suite model failed: {model_id}" in captured.err
    assert f"Suite model '{model_id}' failed" in captured.err
    assert secret not in captured.out
    assert secret not in captured.err

    suite_dirs = list((tmp_path / "suite_runs").iterdir())
    assert len(suite_dirs) == 1
    suite_summary = json.loads(
        (suite_dirs[0] / "suite_summary.json").read_text(encoding="utf-8")
    )
    assert suite_summary["models_completed"] == []
    assert suite_summary["models_failed"] == [model_id]
    assert suite_summary["failure_reasons"] == {
        model_id: "OpenRouter Chat Completions request was rejected."
    }


def test_run_suite_continue_on_error_runs_later_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    failed_model_id = "openrouter/model-a"
    completed_model_id = "openrouter/model-b"
    _write_suite_config(suite_path, [failed_model_id, completed_model_id])
    secret = "test-secret-key-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    credit_values = iter(
        [
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2")),
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2.1")),
            OpenRouterCredits(total_credits=Decimal("10"), total_usage=Decimal("2.2")),
        ]
    )

    def fake_fetch(self):
        return next(credit_values)

    count_order: list[str] = []

    def fake_count(self, text, model=None):
        count_order.append(model.id)
        if model.id == failed_model_id:
            raise CounterRequestError("Provider returned error.")
        return TokenCountResult(
            token_count=3,
            counter="openrouter-usage",
            counting_method="openrouter_usage",
            model_id=model.id,
            tokenizer_name=None,
        )

    monkeypatch.setattr(OpenRouterCreditsClient, "fetch", fake_fetch)
    monkeypatch.setattr(OpenRouterUsageCounter, "count", fake_count)

    exit_code = main(
        [
            "run-suite",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--text-id",
            "short_instruction",
            "--limit",
            "1",
            "--continue-on-error",
            "--yes",
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert count_order == [failed_model_id, completed_model_id]
    assert f"Suite model failed: {failed_model_id}" in captured.err
    assert "Continuing because --continue-on-error was specified." in captured.err
    assert secret not in captured.out
    assert secret not in captured.err

    suite_dirs = list((tmp_path / "suite_runs").iterdir())
    assert len(suite_dirs) == 1
    suite_summary = json.loads(
        (suite_dirs[0] / "suite_summary.json").read_text(encoding="utf-8")
    )
    assert suite_summary["models_completed"] == [completed_model_id]
    assert suite_summary["models_failed"] == [failed_model_id]
    assert suite_summary["failure_reasons"] == {
        failed_model_id: "Provider returned error."
    }
    assert suite_summary["total_rows_executed"] == 1
