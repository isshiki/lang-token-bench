from __future__ import annotations

from pathlib import Path

from lang_token_bench.schema import BenchmarkResult


def write_markdown_report(results: list[BenchmarkResult], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _render_markdown(results)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _render_markdown(results: list[BenchmarkResult]) -> list[str]:
    timestamp = results[0].timestamp_utc if results else ""
    lines = [
        "# Language Token Efficiency Benchmark Results",
        "",
        f"Generated UTC: {timestamp}",
        "",
        "This report compares token counts for aligned multilingual text samples.",
        "OpenRouter usage backends, when added, should be interpreted as observed usage values rather than official tokenizer-only results.",
        "",
        "| Model | Text ID | Language | Tokens | Ratio to English | Counter | Counting Method | Estimated Input Cost USD |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: |",
    ]

    for result in results:
        lines.append(
            "| {model} | {text_id} | {language} | {tokens} | {ratio} | {counter} | {method} | {cost} |".format(
                model=result.model_id,
                text_id=result.text_id,
                language=f"{result.language_name} ({result.language_code})",
                tokens=result.token_count,
                ratio=_format_optional_float(result.ratio_to_english),
                counter=result.counter,
                method=result.counting_method,
                cost=_format_optional_float(result.estimated_input_cost_usd),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `ratio_to_english` uses English token count as 1.0 for each model and text record.",
            "- Empty cost cells mean no input price was configured for the model.",
            "- Results are sensitive to translation choices, text length, terminology, and tokenizer behavior.",
        ]
    )
    return lines


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")

