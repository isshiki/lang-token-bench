from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from lang_token_bench.config import DEFAULT_LANGUAGES_PATH, load_languages
from lang_token_bench.schema import BenchmarkResult, BenchmarkSuiteConfig


@dataclass(frozen=True)
class SummaryRow:
    language_code: str
    language_name: str
    ratios_by_model: dict[str, float | None]
    token_counts_by_model: dict[str, float | None]


@dataclass(frozen=True)
class SummaryTable:
    suite_name: str
    model_ids: list[str]
    rows: list[SummaryRow]


@dataclass(frozen=True)
class SummaryReportPaths:
    latest_csv: Path
    latest_markdown: Path
    latest_heatmap_csv: Path
    latest_token_count_csv: Path
    latest_token_count_markdown: Path
    latest_token_count_heatmap_csv: Path
    suite_csv: Path
    suite_markdown: Path
    suite_heatmap_csv: Path
    suite_token_count_csv: Path
    suite_token_count_markdown: Path
    suite_token_count_heatmap_csv: Path


@dataclass(frozen=True)
class ResultSource:
    run_id: str
    path: Path
    rows_count: int
    timestamp_start_utc: str | None
    timestamp_end_utc: str | None
    model_ids: list[str]


@dataclass(frozen=True)
class SourcedBenchmarkResult:
    result: BenchmarkResult
    source: ResultSource


@dataclass(frozen=True)
class SummarySourceInfo:
    model_id: str
    run_id: str
    path: Path
    adopted_rows_count: int
    source_rows_count: int
    timestamp_start_utc: str | None
    timestamp_end_utc: str | None
    source_model_ids: list[str]


@dataclass(frozen=True)
class ModelResultComparison:
    first_model_id: str
    second_model_id: str
    token_counts_identical: bool
    separate_sources: bool
    first_run_ids: list[str]
    second_run_ids: list[str]
    first_paths: list[Path]
    second_paths: list[Path]


AVG_LABEL = "Avg"


def summarize_suite_results(
    *,
    suite: BenchmarkSuiteConfig,
    output_dir: Path,
    languages_path: Path = DEFAULT_LANGUAGES_PATH,
) -> SummaryTable:
    sourced_results = load_saved_results_with_sources(output_dir)
    if not sourced_results:
        raise ValueError(
            f"No saved benchmark results found under {output_dir}. "
            "Run a benchmark first."
        )

    latest_sourced_results = _latest_sourced_results_by_model_text_language(sourced_results)
    latest_results = [sourced.result for sourced in latest_sourced_results]
    selected = [
        result
        for result in latest_results
        if result.model_id in suite.model_ids and result.ratio_to_english is not None
    ]
    if not selected:
        raise ValueError(f"No saved results found for suite '{suite.name}'.")

    languages = [language for language in load_languages(languages_path) if language.enabled]
    ratio_grouped: dict[tuple[str, str], list[float]] = {}
    token_count_grouped: dict[tuple[str, str], list[int]] = {}
    for result in selected:
        ratio_grouped.setdefault((result.language_code, result.model_id), []).append(
            float(result.ratio_to_english)
        )
        token_count_grouped.setdefault((result.language_code, result.model_id), []).append(
            result.token_count
        )

    rows: list[SummaryRow] = []
    for language in languages:
        ratios_by_model: dict[str, float | None] = {}
        token_counts_by_model: dict[str, float | None] = {}
        for model_id in suite.model_ids:
            values = ratio_grouped.get((language.code, model_id), [])
            if not values:
                ratios_by_model[model_id] = None
            elif language.code == "en":
                ratios_by_model[model_id] = 1.0
            else:
                ratios_by_model[model_id] = round(sum(values) / len(values), 6)
            token_counts = token_count_grouped.get((language.code, model_id), [])
            token_counts_by_model[model_id] = _average_token_counts(token_counts)
        rows.append(
            SummaryRow(
                language_code=language.code,
                language_name=language.name,
                ratios_by_model=ratios_by_model,
                token_counts_by_model=token_counts_by_model,
            )
        )

    return SummaryTable(suite_name=suite.name, model_ids=suite.model_ids, rows=rows)


def load_saved_results(output_dir: Path) -> list[BenchmarkResult]:
    return [sourced.result for sourced in load_saved_results_with_sources(output_dir)]


def load_saved_results_with_sources(output_dir: Path) -> list[SourcedBenchmarkResult]:
    result_paths = sorted((output_dir / "runs").glob("*/results.csv"))
    if (output_dir / "results.csv").exists():
        result_paths.append(output_dir / "results.csv")

    results: list[SourcedBenchmarkResult] = []
    for path in result_paths:
        path_results = load_results_csv(path)
        source = _build_result_source(path, output_dir, path_results)
        results.extend(
            SourcedBenchmarkResult(result=result, source=source)
            for result in path_results
        )
    return results


def load_results_csv(path: Path) -> list[BenchmarkResult]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [_result_from_row(row) for row in reader]


def get_summary_source_info(
    *,
    suite: BenchmarkSuiteConfig,
    output_dir: Path,
) -> list[SummarySourceInfo]:
    sourced_results = load_saved_results_with_sources(output_dir)
    latest_sourced_results = _latest_sourced_results_by_model_text_language(sourced_results)
    selected = [
        sourced
        for sourced in latest_sourced_results
        if sourced.result.model_id in suite.model_ids
        and sourced.result.ratio_to_english is not None
    ]
    grouped: dict[tuple[str, str, Path], list[SourcedBenchmarkResult]] = {}
    for sourced in selected:
        key = (
            sourced.result.model_id,
            sourced.source.run_id,
            sourced.source.path,
        )
        grouped.setdefault(key, []).append(sourced)

    source_info: list[SummarySourceInfo] = []
    for model_id in suite.model_ids:
        model_groups = [
            (key, values)
            for key, values in grouped.items()
            if key[0] == model_id
        ]
        for (_model_id, run_id, path), values in sorted(
            model_groups,
            key=lambda item: (item[0][1], str(item[0][2])),
        ):
            source = values[0].source
            timestamps = [
                sourced.result.timestamp_utc
                for sourced in values
                if sourced.result.timestamp_utc
            ]
            source_info.append(
                SummarySourceInfo(
                    model_id=model_id,
                    run_id=run_id,
                    path=path,
                    adopted_rows_count=len(values),
                    source_rows_count=source.rows_count,
                    timestamp_start_utc=min(timestamps) if timestamps else None,
                    timestamp_end_utc=max(timestamps) if timestamps else None,
                    source_model_ids=source.model_ids,
                )
            )
    return source_info


def compare_model_results(
    *,
    output_dir: Path,
    first_model_id: str,
    second_model_id: str,
) -> ModelResultComparison:
    latest_sourced_results = _latest_sourced_results_by_model_text_language(
        load_saved_results_with_sources(output_dir)
    )
    first_results = _results_for_model(latest_sourced_results, first_model_id)
    second_results = _results_for_model(latest_sourced_results, second_model_id)
    first_counts = {
        key: sourced.result.token_count
        for key, sourced in first_results.items()
    }
    second_counts = {
        key: sourced.result.token_count
        for key, sourced in second_results.items()
    }
    first_paths = _unique_paths(sourced.source.path for sourced in first_results.values())
    second_paths = _unique_paths(sourced.source.path for sourced in second_results.values())
    first_run_ids = _unique_strings(
        sourced.source.run_id for sourced in first_results.values()
    )
    second_run_ids = _unique_strings(
        sourced.source.run_id for sourced in second_results.values()
    )
    return ModelResultComparison(
        first_model_id=first_model_id,
        second_model_id=second_model_id,
        token_counts_identical=first_counts == second_counts,
        separate_sources=bool(first_paths)
        and bool(second_paths)
        and set(first_paths).isdisjoint(second_paths),
        first_run_ids=first_run_ids,
        second_run_ids=second_run_ids,
        first_paths=first_paths,
        second_paths=second_paths,
    )


def write_summary_reports(summary: SummaryTable, output_dir: Path) -> SummaryReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary_ratio_by_language_model.csv"
    md_path = output_dir / "summary_ratio_by_language_model.md"
    heatmap_path = output_dir / "heatmap_ratio_language_model.csv"
    token_count_csv_path = output_dir / "summary_token_count_by_language_model.csv"
    token_count_md_path = output_dir / "summary_token_count_by_language_model.md"
    token_count_heatmap_path = output_dir / "heatmap_token_count_language_model.csv"
    suite_dir = output_dir / "summaries" / safe_summary_suite_name(summary.suite_name)
    suite_csv_path = suite_dir / "summary_ratio_by_language_model.csv"
    suite_md_path = suite_dir / "summary_ratio_by_language_model.md"
    suite_heatmap_path = suite_dir / "heatmap_ratio_language_model.csv"
    suite_token_count_csv_path = suite_dir / "summary_token_count_by_language_model.csv"
    suite_token_count_md_path = suite_dir / "summary_token_count_by_language_model.md"
    suite_token_count_heatmap_path = suite_dir / "heatmap_token_count_language_model.csv"

    _write_summary_csv(summary, csv_path)
    _write_summary_markdown(summary, md_path)
    _write_heatmap_csv(summary, heatmap_path)
    _write_token_count_summary_csv(summary, token_count_csv_path)
    _write_token_count_summary_markdown(summary, token_count_md_path)
    _write_token_count_heatmap_csv(summary, token_count_heatmap_path)
    _write_summary_csv(summary, suite_csv_path)
    _write_summary_markdown(summary, suite_md_path)
    _write_heatmap_csv(summary, suite_heatmap_path)
    _write_token_count_summary_csv(summary, suite_token_count_csv_path)
    _write_token_count_summary_markdown(summary, suite_token_count_md_path)
    _write_token_count_heatmap_csv(summary, suite_token_count_heatmap_path)
    return SummaryReportPaths(
        latest_csv=csv_path,
        latest_markdown=md_path,
        latest_heatmap_csv=heatmap_path,
        latest_token_count_csv=token_count_csv_path,
        latest_token_count_markdown=token_count_md_path,
        latest_token_count_heatmap_csv=token_count_heatmap_path,
        suite_csv=suite_csv_path,
        suite_markdown=suite_md_path,
        suite_heatmap_csv=suite_heatmap_path,
        suite_token_count_csv=suite_token_count_csv_path,
        suite_token_count_markdown=suite_token_count_md_path,
        suite_token_count_heatmap_csv=suite_token_count_heatmap_path,
    )


def safe_summary_suite_name(suite_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", suite_name.strip())
    safe = safe.strip("-._")
    return safe or "unknown"


def _write_summary_csv(summary: SummaryTable, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["language_code", "language_name", *summary.model_ids, AVG_LABEL]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(
                {
                    "language_code": row.language_code,
                    "language_name": row.language_name,
                    **{
                        model_id: _format_optional_ratio(row.ratios_by_model[model_id])
                        for model_id in summary.model_ids
                    },
                    AVG_LABEL: _format_optional_ratio(_row_average(summary, row)),
                }
            )
        writer.writerow(_average_summary_row(summary))
    return path


def _write_token_count_summary_csv(summary: SummaryTable, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["language_code", "language_name", *summary.model_ids, AVG_LABEL]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(
                {
                    "language_code": row.language_code,
                    "language_name": row.language_name,
                    **{
                        model_id: _format_optional_token_count(
                            row.token_counts_by_model[model_id]
                        )
                        for model_id in summary.model_ids
                    },
                    AVG_LABEL: _format_optional_token_count(
                        _token_count_row_average(summary, row)
                    ),
                }
            )
        writer.writerow(_average_token_count_summary_row(summary))
    return path


def _write_heatmap_csv(summary: SummaryTable, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "language_code",
        "language_name",
        "model_id",
        "ratio_to_english",
        "is_average",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            for model_id in summary.model_ids:
                writer.writerow(
                    {
                        "language_code": row.language_code,
                        "language_name": row.language_name,
                        "model_id": model_id,
                        "ratio_to_english": _format_optional_ratio(
                            row.ratios_by_model[model_id]
                        ),
                        "is_average": "false",
                    }
                )
            writer.writerow(
                {
                    "language_code": row.language_code,
                    "language_name": row.language_name,
                    "model_id": AVG_LABEL,
                    "ratio_to_english": _format_optional_ratio(
                        _row_average(summary, row)
                    ),
                    "is_average": "true",
                }
            )
        for model_id in summary.model_ids:
            writer.writerow(
                {
                    "language_code": "avg",
                    "language_name": AVG_LABEL,
                    "model_id": model_id,
                    "ratio_to_english": _format_optional_ratio(
                        _model_average(summary, model_id)
                    ),
                    "is_average": "true",
                }
            )
        writer.writerow(
            {
                "language_code": "avg",
                "language_name": AVG_LABEL,
                "model_id": AVG_LABEL,
                "ratio_to_english": _format_optional_ratio(_overall_average(summary)),
                "is_average": "true",
            }
        )
    return path


def _write_token_count_heatmap_csv(summary: SummaryTable, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "language_code",
        "language_name",
        "model_id",
        "token_count",
        "is_average",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            for model_id in summary.model_ids:
                writer.writerow(
                    {
                        "language_code": row.language_code,
                        "language_name": row.language_name,
                        "model_id": model_id,
                        "token_count": _format_optional_token_count(
                            row.token_counts_by_model[model_id]
                        ),
                        "is_average": "false",
                    }
                )
            writer.writerow(
                {
                    "language_code": row.language_code,
                    "language_name": row.language_name,
                    "model_id": AVG_LABEL,
                    "token_count": _format_optional_token_count(
                        _token_count_row_average(summary, row)
                    ),
                    "is_average": "true",
                }
            )
        for model_id in summary.model_ids:
            writer.writerow(
                {
                    "language_code": "avg",
                    "language_name": AVG_LABEL,
                    "model_id": model_id,
                    "token_count": _format_optional_token_count(
                        _token_count_model_average(summary, model_id)
                    ),
                    "is_average": "true",
                }
            )
        writer.writerow(
            {
                "language_code": "avg",
                "language_name": AVG_LABEL,
                "model_id": AVG_LABEL,
                "token_count": _format_optional_token_count(
                    _token_count_overall_average(summary)
                ),
                "is_average": "true",
            }
        )
    return path


def _write_summary_markdown(summary: SummaryTable, path: Path) -> Path:
    lines = [
        "# Language Token Efficiency Benchmark Summary",
        "",
        f"Suite: `{summary.suite_name}`",
        "",
        "| Language Code | Language Name | "
        + " | ".join(summary.model_ids)
        + f" | {AVG_LABEL} |",
        "| --- | --- | "
        + " | ".join("---:" for _ in [*summary.model_ids, AVG_LABEL])
        + " |",
    ]
    for row in summary.rows:
        lines.append(
            "| {code} | {name} | {values} |".format(
                code=row.language_code,
                name=row.language_name,
                values=" | ".join(
                    [
                        *(
                            _format_markdown_ratio(row.ratios_by_model[model_id])
                            for model_id in summary.model_ids
                        ),
                        _format_markdown_ratio(_row_average(summary, row)),
                    ]
                ),
            )
        )
    lines.append(
        "| {code} | {name} | {values} |".format(
            code="avg",
            name=AVG_LABEL,
            values=" | ".join(
                [
                    *(
                        _format_markdown_ratio(_model_average(summary, model_id))
                        for model_id in summary.model_ids
                    ),
                    _format_markdown_ratio(_overall_average(summary)),
                ]
            ),
        )
    )
    lines.extend(
        [
            "",
            "Values are average `ratio_to_english` by language and model across saved text records.",
            "English is shown as 1.0 when results are available.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_token_count_summary_markdown(summary: SummaryTable, path: Path) -> Path:
    lines = [
        "# Language Token Efficiency Benchmark Token Count Summary",
        "",
        f"Suite: `{summary.suite_name}`",
        "",
        "| Language Code | Language Name | "
        + " | ".join(summary.model_ids)
        + f" | {AVG_LABEL} |",
        "| --- | --- | "
        + " | ".join("---:" for _ in [*summary.model_ids, AVG_LABEL])
        + " |",
    ]
    for row in summary.rows:
        lines.append(
            "| {code} | {name} | {values} |".format(
                code=row.language_code,
                name=row.language_name,
                values=" | ".join(
                    [
                        *(
                            _format_markdown_token_count(
                                row.token_counts_by_model[model_id]
                            )
                            for model_id in summary.model_ids
                        ),
                        _format_markdown_token_count(
                            _token_count_row_average(summary, row)
                        ),
                    ]
                ),
            )
        )
    lines.append(
        "| {code} | {name} | {values} |".format(
            code="avg",
            name=AVG_LABEL,
            values=" | ".join(
                [
                    *(
                        _format_markdown_token_count(
                            _token_count_model_average(summary, model_id)
                        )
                        for model_id in summary.model_ids
                    ),
                    _format_markdown_token_count(_token_count_overall_average(summary)),
                ]
            ),
        )
    )
    lines.extend(
        [
            "",
            "Values are average observed input prompt token counts by language and model across saved text records.",
            "They show absolute prompt size, not the ratio to English.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _latest_results_by_model_text_language(
    results: list[BenchmarkResult],
) -> list[BenchmarkResult]:
    latest: dict[tuple[str, str, str], BenchmarkResult] = {}
    for result in results:
        latest[(result.model_id, result.text_id, result.language_code)] = result
    return list(latest.values())


def _latest_sourced_results_by_model_text_language(
    results: list[SourcedBenchmarkResult],
) -> list[SourcedBenchmarkResult]:
    latest: dict[tuple[str, str, str], SourcedBenchmarkResult] = {}
    for sourced in results:
        result = sourced.result
        latest[(result.model_id, result.text_id, result.language_code)] = sourced
    return list(latest.values())


def _build_result_source(
    path: Path,
    output_dir: Path,
    results: list[BenchmarkResult],
) -> ResultSource:
    timestamps = [
        result.timestamp_utc
        for result in results
        if result.timestamp_utc
    ]
    return ResultSource(
        run_id=_source_run_id(path, output_dir),
        path=path,
        rows_count=len(results),
        timestamp_start_utc=min(timestamps) if timestamps else None,
        timestamp_end_utc=max(timestamps) if timestamps else None,
        model_ids=_unique_strings(result.model_id for result in results),
    )


def _source_run_id(path: Path, output_dir: Path) -> str:
    try:
        relative = path.relative_to(output_dir)
    except ValueError:
        return path.parent.name
    parts = relative.parts
    if parts == ("results.csv",):
        return "latest"
    if len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "results.csv":
        return parts[1]
    return path.parent.name


def _results_for_model(
    results: list[SourcedBenchmarkResult],
    model_id: str,
) -> dict[tuple[str, str], SourcedBenchmarkResult]:
    return {
        (sourced.result.text_id, sourced.result.language_code): sourced
        for sourced in results
        if sourced.result.model_id == model_id
    }


def _unique_strings(values) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _unique_paths(values) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _result_from_row(row: dict[str, str]) -> BenchmarkResult:
    return BenchmarkResult(
        model_id=row["model_id"],
        provider=row["provider"],
        counter=row["counter"],
        counting_method=row["counting_method"],
        language_code=row["language_code"],
        language_name=row["language_name"],
        text_id=row["text_id"],
        token_count=int(row["token_count"]),
        ratio_to_english=_optional_float(row.get("ratio_to_english", "")),
        input_price_per_1m_tokens=_optional_float(row.get("input_price_per_1m_tokens", "")),
        estimated_input_cost_usd=_optional_float(row.get("estimated_input_cost_usd", "")),
        timestamp_utc=row["timestamp_utc"],
    )


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _format_optional_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_markdown_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}x"


def _format_optional_token_count(value: float | None) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_markdown_token_count(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:,.0f}"


def _average_summary_row(summary: SummaryTable) -> dict[str, str]:
    return {
        "language_code": "avg",
        "language_name": AVG_LABEL,
        **{
            model_id: _format_optional_ratio(_model_average(summary, model_id))
            for model_id in summary.model_ids
        },
        AVG_LABEL: _format_optional_ratio(_overall_average(summary)),
    }


def _average_token_count_summary_row(summary: SummaryTable) -> dict[str, str]:
    return {
        "language_code": "avg",
        "language_name": AVG_LABEL,
        **{
            model_id: _format_optional_token_count(
                _token_count_model_average(summary, model_id)
            )
            for model_id in summary.model_ids
        },
        AVG_LABEL: _format_optional_token_count(_token_count_overall_average(summary)),
    }


def _row_average(summary: SummaryTable, row: SummaryRow) -> float | None:
    if row.language_code == "en":
        return 1.0
    values = [
        row.ratios_by_model[model_id]
        for model_id in summary.model_ids
        if row.ratios_by_model[model_id] is not None
    ]
    return _average(values)


def _token_count_row_average(summary: SummaryTable, row: SummaryRow) -> float | None:
    values = [
        row.token_counts_by_model[model_id]
        for model_id in summary.model_ids
        if row.token_counts_by_model[model_id] is not None
    ]
    return _average(values)


def _token_count_model_average(summary: SummaryTable, model_id: str) -> float | None:
    values = [
        row.token_counts_by_model[model_id]
        for row in summary.rows
        if row.token_counts_by_model[model_id] is not None
    ]
    return _average(values)


def _token_count_overall_average(summary: SummaryTable) -> float | None:
    values = [
        row.token_counts_by_model[model_id]
        for row in summary.rows
        for model_id in summary.model_ids
        if row.token_counts_by_model[model_id] is not None
    ]
    return _average(values)


def _model_average(summary: SummaryTable, model_id: str) -> float | None:
    values = [
        row.ratios_by_model[model_id]
        for row in summary.rows
        if row.language_code != "en" and row.ratios_by_model[model_id] is not None
    ]
    return _average(values)


def _overall_average(summary: SummaryTable) -> float | None:
    values = [
        row.ratios_by_model[model_id]
        for row in summary.rows
        for model_id in summary.model_ids
        if row.language_code != "en" and row.ratios_by_model[model_id] is not None
    ]
    return _average(values)


def _average(values: list[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _average_token_counts(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)
