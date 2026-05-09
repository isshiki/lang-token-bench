from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from lang_token_bench.config import (
    DEFAULT_LANGUAGES_PATH,
    DEFAULT_MODELS_PATH,
    load_languages,
    load_models,
)
from lang_token_bench.counters.base import CounterUnavailableError
from lang_token_bench.schema import BenchmarkSuiteConfig, ChartConfig
from lang_token_bench.summary import safe_summary_suite_name


PLOT_INSTALL_MESSAGE = (
    "plot command requires the optional dependency. "
    "Install it with: uv sync --extra viz"
)


@dataclass(frozen=True)
class PlotOutput:
    label: str
    png_path: Path
    svg_path: Path
    markdown_heading: str
    alt_text: str


@dataclass(frozen=True)
class SummaryCsv:
    model_ids: list[str]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class PlotLabels:
    model_labels: dict[str, str]
    language_labels: dict[str, str]


def plot_suite_figures(
    *,
    suite: BenchmarkSuiteConfig,
    output_dir: Path,
    models_path: Path = DEFAULT_MODELS_PATH,
    languages_path: Path = DEFAULT_LANGUAGES_PATH,
) -> list[PlotOutput]:
    summary_dir = output_dir / "summaries" / safe_summary_suite_name(suite.name)
    summary_csv_path = summary_dir / "summary_ratio_by_language_model.csv"
    token_count_summary_csv_path = summary_dir / "summary_token_count_by_language_model.csv"
    if not summary_csv_path.exists():
        raise ValueError(
            f"Summary CSV not found: {summary_csv_path}. "
            f"Run: uv run lang-token-bench summarize --suite {suite.name}"
        )
    if not token_count_summary_csv_path.exists():
        raise ValueError(
            f"Token count summary CSV not found: {token_count_summary_csv_path}. "
            f"Run: uv run lang-token-bench summarize --suite {suite.name}"
        )

    summary_csv = load_summary_csv(summary_csv_path)
    token_count_summary_csv = load_summary_csv(token_count_summary_csv_path)
    labels = build_plot_labels(models_path=models_path, languages_path=languages_path)
    figures_dir = summary_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        _plot_heatmap(
            summary_csv=summary_csv,
            labels=labels,
            figures_dir=figures_dir,
        ),
        _plot_token_count_heatmap(
            summary_csv=token_count_summary_csv,
            labels=labels,
            figures_dir=figures_dir,
        )
    ]
    for chart in suite.charts:
        if chart.type == "two_model_bar":
            outputs.append(
                _plot_two_model_bar(
                    summary_csv=summary_csv,
                    chart=chart,
                    labels=labels,
                    figures_dir=figures_dir,
                )
            )
    _update_summary_markdown_files(
        outputs=outputs,
        output_dir=output_dir,
        suite_name=suite.name,
    )
    return outputs


def build_plot_labels(
    *,
    models_path: Path = DEFAULT_MODELS_PATH,
    languages_path: Path = DEFAULT_LANGUAGES_PATH,
) -> PlotLabels:
    model_labels = {
        model.id: model.short_name or model.display_name or model.id
        for model in load_models(models_path)
    }
    language_labels = {
        language.code: language.plot_label or language.name
        for language in load_languages(languages_path)
    }
    return PlotLabels(model_labels=model_labels, language_labels=language_labels)


def load_summary_csv(path: Path) -> SummaryCsv:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 3:
            raise ValueError(f"Summary CSV has too few columns: {path}")
        model_ids = [
            fieldname
            for fieldname in fieldnames[2:]
            if fieldname != "Avg"
        ]
        return SummaryCsv(model_ids=model_ids, rows=list(reader))


def _plot_heatmap(
    *,
    summary_csv: SummaryCsv,
    labels: PlotLabels,
    figures_dir: Path,
) -> PlotOutput:
    plt = _load_pyplot()
    model_ids = [*summary_csv.model_ids, "Avg"]
    model_labels = [_format_model_label(model_id, labels) for model_id in model_ids]
    language_labels = [
        _format_language_label(row, labels)
        for row in summary_csv.rows
    ]
    values = [
        [_parse_ratio(row.get(model_id, "")) for model_id in model_ids]
        for row in summary_csv.rows
    ]

    fig_width = max(9.0, 1.25 * len(model_ids) + 2.5)
    fig_height = max(5.5, 0.55 * len(language_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap, norm = build_ratio_colormap_and_norm(values)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    fig.colorbar(image, ax=ax, label="Ratio to English")

    ax.set_xticks(range(len(model_ids)))
    ax.set_xticklabels(model_labels, rotation=35, ha="left")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_yticks(range(len(language_labels)))
    ax.set_yticklabels(language_labels)

    for row_index, row_values in enumerate(values):
        for col_index, value in enumerate(row_values):
            if math.isnan(value):
                label = ""
            else:
                label = f"{value:.2f}x"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)

    return _save_figure(
        fig=fig,
        figures_dir=figures_dir,
        output_name="heatmap_ratio_by_language_model",
        label="heatmap",
        markdown_heading="Language x Model Token Ratio",
        alt_text="Language x Model Token Ratio",
        plt=plt,
    )


def _plot_token_count_heatmap(
    *,
    summary_csv: SummaryCsv,
    labels: PlotLabels,
    figures_dir: Path,
) -> PlotOutput:
    plt = _load_pyplot()
    model_ids = [*summary_csv.model_ids, "Avg"]
    model_labels = [_format_model_label(model_id, labels) for model_id in model_ids]
    language_labels = [
        _format_language_label(row, labels)
        for row in summary_csv.rows
    ]
    values = [
        [_parse_ratio(row.get(model_id, "")) for model_id in model_ids]
        for row in summary_csv.rows
    ]

    fig_width = max(9.0, 1.25 * len(model_ids) + 2.5)
    fig_height = max(5.5, 0.55 * len(language_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=build_token_count_colormap(), aspect="auto")
    fig.colorbar(image, ax=ax, label="Input prompt tokens")

    ax.set_xticks(range(len(model_ids)))
    ax.set_xticklabels(model_labels, rotation=35, ha="left")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_yticks(range(len(language_labels)))
    ax.set_yticklabels(language_labels)

    for row_index, row_values in enumerate(values):
        for col_index, value in enumerate(row_values):
            label = "" if math.isnan(value) else f"{value:,.0f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)

    return _save_figure(
        fig=fig,
        figures_dir=figures_dir,
        output_name="heatmap_token_count_by_language_model",
        label="token count heatmap",
        markdown_heading="Language x Model Input Token Count",
        alt_text="Language x Model Input Token Count",
        plt=plt,
    )


def build_token_count_colormap():
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise CounterUnavailableError(PLOT_INSTALL_MESSAGE) from exc

    return LinearSegmentedColormap.from_list(
        "token_count_soft_blue_green",
        [
            "#F7FCF0",
            "#E0F3DB",
            "#CCEBC5",
            "#A8DDB5",
            "#7BCCC4",
            "#4EB3D3",
            "#5A9FCD",
        ],
    )


def _plot_two_model_bar(
    *,
    summary_csv: SummaryCsv,
    chart: ChartConfig,
    labels: PlotLabels,
    figures_dir: Path,
) -> PlotOutput:
    if len(chart.model_ids) != 2:
        raise ValueError(f"Chart '{chart.id}' must contain exactly two model_ids.")
    for model_id in chart.model_ids:
        if model_id not in summary_csv.model_ids:
            raise ValueError(f"Chart '{chart.id}' model_id not found in summary CSV: {model_id}")

    plt = _load_pyplot()
    rows = sort_bar_rows(
        [row for row in summary_csv.rows if row.get("language_code") != "avg"],
        chart.model_ids,
        sort_method=chart.sort_languages,
    )
    languages = [_format_language_label(row, labels) for row in rows]
    first_values = [_parse_ratio(row.get(chart.model_ids[0], "")) for row in rows]
    second_values = [_parse_ratio(row.get(chart.model_ids[1], "")) for row in rows]
    positions = list(range(len(rows)))
    bar_width = 0.38

    fig_width = max(9.0, 0.85 * len(languages) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    first_bars = ax.bar(
        [position - bar_width / 2 for position in positions],
        [_finite_or_zero(value) for value in first_values],
        width=bar_width,
        label=_format_model_label(chart.model_ids[0], labels),
    )
    second_bars = ax.bar(
        [position + bar_width / 2 for position in positions],
        [_finite_or_zero(value) for value in second_values],
        width=bar_width,
        label=_format_model_label(chart.model_ids[1], labels),
    )
    ax.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_ylabel("Ratio to English")
    ax.set_xticks(positions)
    ax.set_xticklabels(languages, rotation=25, ha="right")
    _place_legend(ax, chart.legend_position)
    if chart.show_value_labels:
        _add_bar_value_labels(ax, first_bars, first_values)
        _add_bar_value_labels(ax, second_bars, second_values)
    finite_values = [
        value
        for value in [*first_values, *second_values]
        if not math.isnan(value)
    ]
    if finite_values:
        ax.set_ylim(0, max(finite_values) * 1.18)

    return _save_figure(
        fig=fig,
        figures_dir=figures_dir,
        output_name=safe_summary_suite_name(chart.output_name),
        label=chart.id,
        markdown_heading=_format_chart_heading(chart),
        alt_text=chart.title,
        plt=plt,
    )


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.fonttype"] = "none"
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise CounterUnavailableError(PLOT_INSTALL_MESSAGE) from exc
    return plt


def build_ratio_colormap_and_norm(values: list[list[float]]):
    try:
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError as exc:
        raise CounterUnavailableError(PLOT_INSTALL_MESSAGE) from exc

    numeric_values = [
        value
        for row in values
        for value in row
        if not math.isnan(value)
    ]
    data_min = min(numeric_values) if numeric_values else 1.0
    data_max = max(numeric_values) if numeric_values else 1.0
    vmin = min(0.5, data_min)
    vmax = max(2.0, data_max)
    cmap = LinearSegmentedColormap.from_list(
        "token_efficiency_green_white_orange",
        [
            (0.0, "#1B7837"),
            (0.5, "#F7F7F7"),
            (1.0, "#F28E2B"),
        ],
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    return cmap, norm


def sort_bar_rows(
    rows: list[dict[str, str]],
    model_ids: list[str],
    *,
    sort_method: str = "by_lower_average_model",
) -> list[dict[str, str]]:
    if sort_method != "by_lower_average_model" or len(model_ids) != 2:
        return rows

    first_model_average = _model_average_from_rows(rows, model_ids[0])
    second_model_average = _model_average_from_rows(rows, model_ids[1])
    if first_model_average is None and second_model_average is None:
        baseline_model_id = model_ids[0]
    elif first_model_average is None:
        baseline_model_id = model_ids[1]
    elif second_model_average is None:
        baseline_model_id = model_ids[0]
    elif first_model_average <= second_model_average:
        baseline_model_id = model_ids[0]
    else:
        baseline_model_id = model_ids[1]

    original_order = {
        row.get("language_code", ""): index
        for index, row in enumerate(rows)
    }
    return sorted(
        rows,
        key=lambda row: (
            _sort_ratio_value(row.get(baseline_model_id, "")),
            original_order.get(row.get("language_code", ""), len(rows)),
        ),
    )


def _save_figure(
    *,
    fig,
    figures_dir: Path,
    output_name: str,
    label: str,
    markdown_heading: str,
    alt_text: str,
    plt,
) -> PlotOutput:
    png_path = figures_dir / f"{output_name}.png"
    svg_path = figures_dir / f"{output_name}.svg"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return PlotOutput(
        label=label,
        png_path=png_path,
        svg_path=svg_path,
        markdown_heading=markdown_heading,
        alt_text=alt_text,
    )


def _format_model_label(model_id: str, labels: PlotLabels) -> str:
    if model_id == "Avg":
        return "Avg"
    return labels.model_labels.get(model_id, model_id)


def _format_language_label(row: dict[str, str], labels: PlotLabels) -> str:
    code = row.get("language_code", "")
    name = row.get("language_name", "")
    if code == "avg":
        return "Avg"
    return labels.language_labels.get(code, name or code)


def _parse_ratio(value: str) -> float:
    if value == "":
        return math.nan
    return float(value)


def _finite_or_zero(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return value


def _model_average_from_rows(rows: list[dict[str, str]], model_id: str) -> float | None:
    values = [
        _parse_ratio(row.get(model_id, ""))
        for row in rows
        if row.get("language_code") != "en"
    ]
    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _sort_ratio_value(value: str) -> float:
    ratio = _parse_ratio(value)
    if math.isnan(ratio):
        return math.inf
    return ratio


def _place_legend(ax, legend_position: str) -> None:
    if legend_position == "top":
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
        )
        return
    ax.legend(frameon=False)


def _add_bar_value_labels(ax, bars, values: list[float]) -> None:
    for bar, value in zip(bars, values, strict=True):
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )


def _format_chart_heading(chart: ChartConfig) -> str:
    if chart.id == "openai_vs_anthropic_bar":
        return "OpenAI vs Anthropic"
    return chart.title


def _update_summary_markdown_files(
    *,
    outputs: list[PlotOutput],
    output_dir: Path,
    suite_name: str,
) -> None:
    safe_suite_name = safe_summary_suite_name(suite_name)
    suite_md_path = (
        output_dir
        / "summaries"
        / safe_suite_name
        / "summary_ratio_by_language_model.md"
    )
    if suite_md_path.exists():
        _upsert_figures_section(
            suite_md_path,
            _build_figures_section(outputs, path_prefix="figures"),
        )

    suite_token_count_md_path = (
        output_dir
        / "summaries"
        / safe_suite_name
        / "summary_token_count_by_language_model.md"
    )
    if suite_token_count_md_path.exists():
        _upsert_figures_section(
            suite_token_count_md_path,
            _build_figures_section(outputs, path_prefix="figures"),
        )

    latest_md_path = output_dir / "summary_ratio_by_language_model.md"
    if latest_md_path.exists():
        _upsert_figures_section(
            latest_md_path,
            _build_figures_section(
                outputs,
                path_prefix=f"summaries/{safe_suite_name}/figures",
            ),
        )

    latest_token_count_md_path = output_dir / "summary_token_count_by_language_model.md"
    if latest_token_count_md_path.exists():
        _upsert_figures_section(
            latest_token_count_md_path,
            _build_figures_section(
                outputs,
                path_prefix=f"summaries/{safe_suite_name}/figures",
            ),
        )


def _build_figures_section(outputs: list[PlotOutput], *, path_prefix: str) -> str:
    lines = ["## Figures", ""]
    for output in outputs:
        image_path = output.svg_path if output.svg_path.exists() else output.png_path
        relative_name = image_path.name
        lines.extend(
            [
                f"### {output.markdown_heading}",
                "",
                f"![{output.alt_text}]({path_prefix}/{relative_name})",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _upsert_figures_section(path: Path, figures_section: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "\n## Figures\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n"
    path.write_text(text + figures_section, encoding="utf-8")
