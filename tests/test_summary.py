from __future__ import annotations

import csv

from lang_token_bench.benchmark import run_benchmark
from lang_token_bench.cli import main
from lang_token_bench.config import load_benchmark_suite
from lang_token_bench.reporters.csv_reporter import write_csv_report
from lang_token_bench.schema import BenchmarkResult
from lang_token_bench.summary import (
    compare_model_results,
    get_summary_source_info,
    safe_summary_suite_name,
)


def _write_suite_config(path) -> None:
    path.write_text(
        "\n".join(
            [
                "suites:",
                "  - name: test_suite",
                "    description: Test suite",
                "    model_ids:",
                "      - simple/baseline",
                "      - openai/gpt-4o-mini",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _benchmark_result(
    *,
    model_id: str,
    text_id: str,
    language_code: str,
    token_count: int,
    timestamp_utc: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        model_id=model_id,
        provider="openrouter",
        counter="openrouter-usage",
        counting_method="openrouter_usage",
        language_code=language_code,
        language_name={"en": "English", "ja": "Japanese"}[language_code],
        text_id=text_id,
        token_count=token_count,
        ratio_to_english=1.0 if language_code == "en" else 1.25,
        input_price_per_1m_tokens=None,
        estimated_input_cost_usd=None,
        timestamp_utc=timestamp_utc,
    )


def test_summarize_command_writes_suite_outputs(tmp_path) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path)
    results = run_benchmark(counter_filter="simple")
    run_dir = tmp_path / "runs" / "20260429T000000_simple-baseline"
    write_csv_report(results, run_dir / "results.csv")

    exit_code = main(
        [
            "summarize",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    summary_csv = tmp_path / "summary_ratio_by_language_model.csv"
    summary_md = tmp_path / "summary_ratio_by_language_model.md"
    heatmap_csv = tmp_path / "heatmap_ratio_language_model.csv"
    token_count_csv = tmp_path / "summary_token_count_by_language_model.csv"
    token_count_md = tmp_path / "summary_token_count_by_language_model.md"
    token_count_heatmap_csv = tmp_path / "heatmap_token_count_language_model.csv"
    relative_token_count_csv = tmp_path / "summary_relative_token_count_by_language_model.csv"
    relative_token_count_md = tmp_path / "summary_relative_token_count_by_language_model.md"
    relative_token_count_heatmap_csv = tmp_path / "heatmap_relative_token_count_language_model.csv"
    weighted_ratio_csv = tmp_path / "summary_weighted_ratio_by_language_model.csv"
    weighted_ratio_md = tmp_path / "summary_weighted_ratio_by_language_model.md"
    weighted_ratio_heatmap_csv = tmp_path / "heatmap_weighted_ratio_language_model.csv"
    excess_tokens_csv = tmp_path / "summary_excess_tokens_by_language_model.csv"
    excess_tokens_md = tmp_path / "summary_excess_tokens_by_language_model.md"
    excess_tokens_heatmap_csv = tmp_path / "heatmap_excess_tokens_language_model.csv"
    suite_summary_dir = tmp_path / "summaries" / "test_suite"
    suite_summary_csv = suite_summary_dir / "summary_ratio_by_language_model.csv"
    suite_summary_md = suite_summary_dir / "summary_ratio_by_language_model.md"
    suite_heatmap_csv = suite_summary_dir / "heatmap_ratio_language_model.csv"
    suite_token_count_csv = suite_summary_dir / "summary_token_count_by_language_model.csv"
    suite_token_count_md = suite_summary_dir / "summary_token_count_by_language_model.md"
    suite_token_count_heatmap_csv = suite_summary_dir / "heatmap_token_count_language_model.csv"
    suite_relative_token_count_csv = suite_summary_dir / "summary_relative_token_count_by_language_model.csv"
    suite_relative_token_count_md = suite_summary_dir / "summary_relative_token_count_by_language_model.md"
    suite_relative_token_count_heatmap_csv = suite_summary_dir / "heatmap_relative_token_count_language_model.csv"
    suite_weighted_ratio_csv = suite_summary_dir / "summary_weighted_ratio_by_language_model.csv"
    suite_weighted_ratio_md = suite_summary_dir / "summary_weighted_ratio_by_language_model.md"
    suite_weighted_ratio_heatmap_csv = suite_summary_dir / "heatmap_weighted_ratio_language_model.csv"
    suite_excess_tokens_csv = suite_summary_dir / "summary_excess_tokens_by_language_model.csv"
    suite_excess_tokens_md = suite_summary_dir / "summary_excess_tokens_by_language_model.md"
    suite_excess_tokens_heatmap_csv = suite_summary_dir / "heatmap_excess_tokens_language_model.csv"
    assert summary_csv.exists()
    assert summary_md.exists()
    assert heatmap_csv.exists()
    assert token_count_csv.exists()
    assert token_count_md.exists()
    assert token_count_heatmap_csv.exists()
    assert relative_token_count_csv.exists()
    assert relative_token_count_md.exists()
    assert relative_token_count_heatmap_csv.exists()
    assert weighted_ratio_csv.exists()
    assert weighted_ratio_md.exists()
    assert weighted_ratio_heatmap_csv.exists()
    assert excess_tokens_csv.exists()
    assert excess_tokens_md.exists()
    assert excess_tokens_heatmap_csv.exists()
    assert suite_summary_csv.exists()
    assert suite_summary_md.exists()
    assert suite_heatmap_csv.exists()
    assert suite_token_count_csv.exists()
    assert suite_token_count_md.exists()
    assert suite_token_count_heatmap_csv.exists()
    assert suite_relative_token_count_csv.exists()
    assert suite_relative_token_count_md.exists()
    assert suite_relative_token_count_heatmap_csv.exists()
    assert suite_weighted_ratio_csv.exists()
    assert suite_weighted_ratio_md.exists()
    assert suite_weighted_ratio_heatmap_csv.exists()
    assert suite_excess_tokens_csv.exists()
    assert suite_excess_tokens_md.exists()
    assert suite_excess_tokens_heatmap_csv.exists()

    with summary_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    english = next(row for row in rows if row["language_code"] == "en")
    avg_row = next(row for row in rows if row["language_code"] == "avg")
    assert english["simple/baseline"] == "1"
    assert english["Avg"] == "1"
    assert avg_row["language_name"] == "Avg"
    assert avg_row["simple/baseline"]
    assert avg_row["Avg"]
    assert "openai/gpt-4o-mini" in english
    assert len(rows) == 9
    summary_text = summary_md.read_text(encoding="utf-8")
    suite_summary_text = suite_summary_md.read_text(encoding="utf-8")
    assert "Language Token Efficiency Benchmark Summary" in summary_text
    assert "Suite: `test_suite`" in summary_text
    assert "| avg | Avg |" in summary_text
    assert "1.00x" in summary_text
    assert "Suite: `test_suite`" in suite_summary_text

    with heatmap_csv.open("r", encoding="utf-8", newline="") as file:
        heatmap_rows = list(csv.DictReader(file))
    assert {
        "language_code",
        "language_name",
        "model_id",
        "ratio_to_english",
        "is_average",
    } == set(heatmap_rows[0])
    assert any(
        row["language_code"] == "avg"
        and row["model_id"] == "Avg"
        and row["is_average"] == "true"
        for row in heatmap_rows
    )

    with token_count_csv.open("r", encoding="utf-8", newline="") as file:
        token_count_rows = list(csv.DictReader(file))
    token_count_english = next(row for row in token_count_rows if row["language_code"] == "en")
    assert token_count_english["simple/baseline"]
    token_count_text = token_count_md.read_text(encoding="utf-8")
    assert "input prompt token counts" in token_count_text

    with token_count_heatmap_csv.open("r", encoding="utf-8", newline="") as file:
        token_count_heatmap_rows = list(csv.DictReader(file))
    assert {
        "language_code",
        "language_name",
        "model_id",
        "token_count",
        "is_average",
    } == set(token_count_heatmap_rows[0])
    assert any(
        row["language_code"] == "avg"
        and row["model_id"] == "Avg"
        and row["is_average"] == "true"
        for row in token_count_heatmap_rows
    )
    assert any(
        row["language_code"] == "en"
        and row["model_id"] == "Avg"
        and row["ratio_to_english"] == "1"
        for row in heatmap_rows
    )

    with relative_token_count_heatmap_csv.open("r", encoding="utf-8", newline="") as file:
        relative_token_count_heatmap_rows = list(csv.DictReader(file))
    assert {
        "language_code",
        "language_name",
        "model_id",
        "relative_token_count",
        "is_average",
    } == set(relative_token_count_heatmap_rows[0])
    assert "minimum cell in this summary table" in relative_token_count_md.read_text(
        encoding="utf-8"
    )

    with weighted_ratio_heatmap_csv.open("r", encoding="utf-8", newline="") as file:
        weighted_ratio_heatmap_rows = list(csv.DictReader(file))
    assert {
        "language_code",
        "language_name",
        "model_id",
        "weighted_ratio_to_english",
        "is_average",
    } == set(weighted_ratio_heatmap_rows[0])
    assert "weighted `ratio_to_english`" in weighted_ratio_md.read_text(
        encoding="utf-8"
    )

    with excess_tokens_heatmap_csv.open("r", encoding="utf-8", newline="") as file:
        excess_tokens_heatmap_rows = list(csv.DictReader(file))
    assert {
        "language_code",
        "language_name",
        "model_id",
        "excess_tokens_vs_english",
        "is_average",
    } == set(excess_tokens_heatmap_rows[0])
    assert "minus the matching English total" in excess_tokens_md.read_text(
        encoding="utf-8"
    )


def test_summarize_command_falls_back_to_latest_results(tmp_path) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    _write_suite_config(suite_path)
    results = run_benchmark(counter_filter="simple")
    write_csv_report(results, tmp_path / "results.csv")

    exit_code = main(
        [
            "summarize",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    assert (tmp_path / "summary_ratio_by_language_model.csv").exists()
    assert (tmp_path / "summary_token_count_by_language_model.csv").exists()
    assert (tmp_path / "summary_relative_token_count_by_language_model.csv").exists()
    assert (tmp_path / "summary_weighted_ratio_by_language_model.csv").exists()
    assert (tmp_path / "summary_excess_tokens_by_language_model.csv").exists()
    assert (
        tmp_path
        / "summaries"
        / "test_suite"
        / "summary_ratio_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / "test_suite"
        / "summary_token_count_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / "test_suite"
        / "summary_relative_token_count_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / "test_suite"
        / "summary_weighted_ratio_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / "test_suite"
        / "summary_excess_tokens_by_language_model.csv"
    ).exists()


def test_summarize_writes_weighted_ratio_and_excess_token_values(tmp_path) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suites:",
                "  - name: test_suite",
                "    description: Test suite",
                "    model_ids:",
                "      - model/a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv_report(
        [
            _benchmark_result(
                model_id="model/a",
                text_id="sample_a",
                language_code="en",
                token_count=10,
                timestamp_utc="2026-04-29T00:00:00Z",
            ),
            _benchmark_result(
                model_id="model/a",
                text_id="sample_a",
                language_code="ja",
                token_count=20,
                timestamp_utc="2026-04-29T00:00:01Z",
            ),
            _benchmark_result(
                model_id="model/a",
                text_id="sample_b",
                language_code="en",
                token_count=20,
                timestamp_utc="2026-04-29T00:00:02Z",
            ),
            _benchmark_result(
                model_id="model/a",
                text_id="sample_b",
                language_code="ja",
                token_count=30,
                timestamp_utc="2026-04-29T00:00:03Z",
            ),
        ],
        tmp_path / "runs" / "20260429T000000_model-a" / "results.csv",
    )

    exit_code = main(
        [
            "summarize",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    with (
        tmp_path / "summary_relative_token_count_by_language_model.csv"
    ).open("r", encoding="utf-8", newline="") as file:
        relative_rows = list(csv.DictReader(file))
    with (
        tmp_path / "summary_weighted_ratio_by_language_model.csv"
    ).open("r", encoding="utf-8", newline="") as file:
        weighted_rows = list(csv.DictReader(file))
    with (
        tmp_path / "summary_excess_tokens_by_language_model.csv"
    ).open("r", encoding="utf-8", newline="") as file:
        excess_rows = list(csv.DictReader(file))

    japanese_relative = next(row for row in relative_rows if row["language_code"] == "ja")
    japanese_weighted = next(row for row in weighted_rows if row["language_code"] == "ja")
    japanese_excess = next(row for row in excess_rows if row["language_code"] == "ja")
    english_excess = next(row for row in excess_rows if row["language_code"] == "en")
    assert japanese_relative["model/a"] == "1.666667"
    assert japanese_weighted["model/a"] == "1.666667"
    assert japanese_excess["model/a"] == "20"
    assert english_excess["model/a"] == "0"


def test_summarize_suite_name_is_sanitized_for_output_path(tmp_path) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suites:",
                "  - name: unsafe/suite:../name",
                "    description: Test suite",
                "    model_ids:",
                "      - simple/baseline",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results = run_benchmark(counter_filter="simple")
    write_csv_report(results, tmp_path / "results.csv")

    exit_code = main(
        [
            "summarize",
            "--suite",
            "unsafe/suite:../name",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    safe_name = safe_summary_suite_name("unsafe/suite:../name")
    assert exit_code == 0
    assert safe_name == "unsafe-suite-..-name"
    assert (
        tmp_path
        / "summaries"
        / safe_name
        / "summary_ratio_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / safe_name
        / "summary_token_count_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / safe_name
        / "summary_relative_token_count_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / safe_name
        / "summary_weighted_ratio_by_language_model.csv"
    ).exists()
    assert (
        tmp_path
        / "summaries"
        / safe_name
        / "summary_excess_tokens_by_language_model.csv"
    ).exists()
    assert not (tmp_path / "summaries" / "unsafe").exists()


def test_summarize_debug_sources_prints_selected_run_details(
    tmp_path,
    capsys,
) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    model_ids = ["model/a", "model/b"]
    suite_path.write_text(
        "\n".join(
            [
                "suites:",
                "  - name: test_suite",
                "    description: Test suite",
                "    model_ids:",
                "      - model/a",
                "      - model/b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for model_id in model_ids:
        run_id = f"20260429T000000_{model_id.replace('/', '-')}"
        write_csv_report(
            [
                _benchmark_result(
                    model_id=model_id,
                    text_id="sample",
                    language_code="en",
                    token_count=10,
                    timestamp_utc="2026-04-29T00:00:00Z",
                ),
                _benchmark_result(
                    model_id=model_id,
                    text_id="sample",
                    language_code="ja",
                    token_count=13,
                    timestamp_utc="2026-04-29T00:01:00Z",
                ),
            ],
            tmp_path / "runs" / run_id / "results.csv",
        )

    exit_code = main(
        [
            "summarize",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
            "--debug-sources",
        ],
        load_env=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Summary source debug:" in captured.out
    assert "Model: model/a" in captured.out
    assert "Model: model/b" in captured.out
    assert "run_id: 20260429T000000_model-a" in captured.out
    assert "run_id: 20260429T000000_model-b" in captured.out
    assert "adopted_rows: 2" in captured.out
    assert "source_rows: 2" in captured.out
    assert "timestamp_range: 2026-04-29T00:00:00Z to 2026-04-29T00:01:00Z" in captured.out
    assert "source_model_ids: model/a" in captured.out
    assert "source_model_ids: model/b" in captured.out

    suite = load_benchmark_suite("test_suite", suite_path)
    source_info = get_summary_source_info(suite=suite, output_dir=tmp_path)
    assert {info.model_id: info.run_id for info in source_info} == {
        "model/a": "20260429T000000_model-a",
        "model/b": "20260429T000000_model-b",
    }


def test_compare_model_results_allows_identical_counts_from_separate_runs(tmp_path) -> None:
    first_model = "model/a"
    second_model = "model/b"
    for model_id in [first_model, second_model]:
        run_id = f"20260429T000000_{model_id.replace('/', '-')}"
        write_csv_report(
            [
                _benchmark_result(
                    model_id=model_id,
                    text_id="sample",
                    language_code="en",
                    token_count=10,
                    timestamp_utc="2026-04-29T00:00:00Z",
                ),
                _benchmark_result(
                    model_id=model_id,
                    text_id="sample",
                    language_code="ja",
                    token_count=13,
                    timestamp_utc="2026-04-29T00:01:00Z",
                ),
            ],
            tmp_path / "runs" / run_id / "results.csv",
        )

    comparison = compare_model_results(
        output_dir=tmp_path,
        first_model_id=first_model,
        second_model_id=second_model,
    )

    assert comparison.token_counts_identical is True
    assert comparison.separate_sources is True
    assert comparison.first_run_ids == ["20260429T000000_model-a"]
    assert comparison.second_run_ids == ["20260429T000000_model-b"]
