from __future__ import annotations

from pathlib import Path

from lang_token_bench.benchmark import run_benchmark
from lang_token_bench.reporters.csv_reporter import write_csv_report
from lang_token_bench.reporters.markdown_reporter import write_markdown_report


def test_run_benchmark_with_simple_counter() -> None:
    results = run_benchmark(counter_filter="simple")

    assert len(results) == 40
    assert {result.counter for result in results} == {"simple"}
    english = next(result for result in results if result.language_code == "en")
    assert english.ratio_to_english == 1.0
    assert {result.text_id for result in results} == {
        "short_instruction",
        "medium_coding_instruction",
        "long_news_summary_instruction",
        "long_system_prompt",
        "very_long_article",
    }


def test_reporters_write_csv_and_markdown(tmp_path: Path) -> None:
    results = run_benchmark(counter_filter="simple")

    csv_path = write_csv_report(results, tmp_path / "results.csv")
    md_path = write_markdown_report(results, tmp_path / "results.md")

    assert csv_path.exists()
    assert md_path.exists()
    assert "ratio_to_english" in csv_path.read_text(encoding="utf-8")
    assert "Language Token Efficiency Benchmark Results" in md_path.read_text(encoding="utf-8")
