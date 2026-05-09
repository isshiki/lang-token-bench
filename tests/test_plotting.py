from __future__ import annotations

import pytest

import lang_token_bench.plotting as plotting
from lang_token_bench.cli import main
from lang_token_bench.plotting import (
    build_plot_labels,
    build_ratio_colormap_and_norm,
    build_token_count_colormap,
    sort_bar_rows,
)


pytest.importorskip("matplotlib")


def _write_summary_csv(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "language_code,language_name,model/a,model/b,Avg",
                "en,English,1,1,1",
                "ja,Japanese,1.4,1.6,1.5",
                "fr,French,1.2,1.3,1.25",
                "avg,Avg,1.3,1.45,1.375",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_token_count_summary_csv(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "language_code,language_name,model/a,model/b,Avg",
                "en,English,100,110,105",
                "ja,Japanese,140,160,150",
                "fr,French,120,130,125",
                "avg,Avg,120,133.33,126.67",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_summary_markdown(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Language Token Efficiency Benchmark Summary\n\nSuite: `test_suite`\n",
        encoding="utf-8",
    )


def _write_models_config(path) -> None:
    path.write_text(
        "\n".join(
            [
                "models:",
                "  - id: model/a",
                "    provider: openrouter",
                "    display_name: Model A Long Name",
                "    short_name: Short A",
                "    counter: openrouter-usage",
                "    tokenizer_name: null",
                "    input_price_per_1m_tokens: null",
                "    enabled: false",
                "  - id: model/b",
                "    provider: openrouter",
                "    display_name: Model B Long Name",
                "    short_name: Short B",
                "    counter: openrouter-usage",
                "    tokenizer_name: null",
                "    input_price_per_1m_tokens: null",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_languages_config(path) -> None:
    path.write_text(
        "\n".join(
            [
                "languages:",
                "  - code: en",
                "    name: English",
                "    plot_label: English Label",
                "    native_name: English",
                "    enabled: true",
                "  - code: ja",
                "    name: Japanese",
                "    plot_label: Japanese Label",
                "    native_name: 日本語",
                "    enabled: true",
                "  - code: fr",
                "    name: French",
                "    plot_label: French Label",
                "    native_name: Français",
                "    enabled: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_plot_command_writes_heatmap_and_configured_bar(tmp_path, monkeypatch) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    models_path = tmp_path / "models.yaml"
    languages_path = tmp_path / "languages.yaml"
    _write_models_config(models_path)
    _write_languages_config(languages_path)
    suite_path.write_text(
        "\n".join(
            [
                "suites:",
                "  - name: test_suite",
                "    description: Test suite",
                "    model_ids:",
                "      - model/a",
                "      - model/b",
                "    charts:",
                "      - id: model_a_vs_b",
                "        type: two_model_bar",
                "        title: Model A vs Model B",
                "        model_ids:",
                "          - model/a",
                "          - model/b",
                "        output_name: model_a_vs_b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_dir = tmp_path / "summaries" / "test_suite"
    _write_summary_csv(summary_dir / "summary_ratio_by_language_model.csv")
    _write_token_count_summary_csv(summary_dir / "summary_token_count_by_language_model.csv")
    _write_summary_markdown(summary_dir / "summary_ratio_by_language_model.md")
    _write_summary_markdown(summary_dir / "summary_token_count_by_language_model.md")
    _write_summary_markdown(tmp_path / "summary_ratio_by_language_model.md")
    _write_summary_markdown(tmp_path / "summary_token_count_by_language_model.md")
    colormap_calls = []
    original_build_colormap = plotting.build_ratio_colormap_and_norm

    def wrapped_build_colormap(values):
        colormap_calls.append(values)
        return original_build_colormap(values)

    monkeypatch.setattr(
        plotting,
        "build_ratio_colormap_and_norm",
        wrapped_build_colormap,
    )

    command = [
            "plot",
            "--suite",
            "test_suite",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
            "--models",
            str(models_path),
            "--languages",
            str(languages_path),
    ]

    exit_code = main(command, load_env=False)
    second_exit_code = main(command, load_env=False)

    figures_dir = summary_dir / "figures"
    assert exit_code == 0
    assert second_exit_code == 0
    assert (figures_dir / "heatmap_ratio_by_language_model.png").exists()
    assert (figures_dir / "heatmap_ratio_by_language_model.svg").exists()
    assert (figures_dir / "heatmap_token_count_by_language_model.png").exists()
    assert (figures_dir / "heatmap_token_count_by_language_model.svg").exists()
    assert (figures_dir / "model_a_vs_b.png").exists()
    assert (figures_dir / "model_a_vs_b.svg").exists()

    heatmap_svg = (figures_dir / "heatmap_ratio_by_language_model.svg").read_text(
        encoding="utf-8"
    )
    bar_svg = (figures_dir / "model_a_vs_b.svg").read_text(encoding="utf-8")
    assert "Short A" in heatmap_svg
    assert "Japanese Label" in heatmap_svg
    assert "Short B" in bar_svg
    assert "Japanese Label" in bar_svg
    assert colormap_calls

    suite_markdown = (
        summary_dir / "summary_ratio_by_language_model.md"
    ).read_text(encoding="utf-8")
    suite_token_count_markdown = (
        summary_dir / "summary_token_count_by_language_model.md"
    ).read_text(encoding="utf-8")
    latest_markdown = (tmp_path / "summary_ratio_by_language_model.md").read_text(
        encoding="utf-8"
    )
    latest_token_count_markdown = (
        tmp_path / "summary_token_count_by_language_model.md"
    ).read_text(encoding="utf-8")
    assert suite_markdown.count("## Figures") == 1
    assert suite_token_count_markdown.count("## Figures") == 1
    assert latest_markdown.count("## Figures") == 1
    assert latest_token_count_markdown.count("## Figures") == 1
    assert "![Language x Model Token Ratio](figures/heatmap_ratio_by_language_model.svg)" in suite_markdown
    assert "![Language x Model Input Token Count](figures/heatmap_token_count_by_language_model.svg)" in suite_markdown
    assert "![Language x Model Input Token Count](figures/heatmap_token_count_by_language_model.svg)" in suite_token_count_markdown
    assert "![Model A vs Model B](figures/model_a_vs_b.svg)" in suite_markdown
    assert (
        "![Language x Model Token Ratio](summaries/test_suite/figures/heatmap_ratio_by_language_model.svg)"
        in latest_markdown
    )
    assert (
        "![Language x Model Input Token Count](summaries/test_suite/figures/heatmap_token_count_by_language_model.svg)"
        in latest_token_count_markdown
    )


def test_plot_command_uses_safe_suite_folder(tmp_path) -> None:
    suite_path = tmp_path / "benchmark_suites.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suites:",
                "  - name: unsafe/suite:../name",
                "    description: Test suite",
                "    model_ids:",
                "      - model/a",
                "      - model/b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_dir = tmp_path / "summaries" / "unsafe-suite-..-name"
    _write_summary_csv(summary_dir / "summary_ratio_by_language_model.csv")
    _write_token_count_summary_csv(summary_dir / "summary_token_count_by_language_model.csv")

    exit_code = main(
        [
            "plot",
            "--suite",
            "unsafe/suite:../name",
            "--suites",
            str(suite_path),
            "--output-dir",
            str(tmp_path),
        ],
        load_env=False,
    )

    assert exit_code == 0
    assert (
        summary_dir
        / "figures"
        / "heatmap_ratio_by_language_model.png"
    ).exists()
    assert (
        summary_dir
        / "figures"
        / "heatmap_token_count_by_language_model.png"
    ).exists()
    assert not (tmp_path / "summaries" / "unsafe").exists()


def test_plot_labels_use_short_name_and_plot_label(tmp_path) -> None:
    models_path = tmp_path / "models.yaml"
    languages_path = tmp_path / "languages.yaml"
    _write_models_config(models_path)
    _write_languages_config(languages_path)

    labels = build_plot_labels(models_path=models_path, languages_path=languages_path)

    assert labels.model_labels["model/a"] == "Short A"
    assert labels.language_labels["ja"] == "Japanese Label"


def test_heatmap_colormap_uses_two_slope_norm_centered_on_english_ratio() -> None:
    from matplotlib.colors import TwoSlopeNorm

    cmap, norm = build_ratio_colormap_and_norm([[0.4, 1.0, 2.3]])

    assert cmap.name == "token_efficiency_green_white_orange"
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vmin == pytest.approx(0.4)
    assert norm.vcenter == pytest.approx(1.0)
    assert norm.vmax == pytest.approx(2.3)


def test_heatmap_colormap_uses_default_ratio_bounds() -> None:
    _, norm = build_ratio_colormap_and_norm([[0.8, 1.0, 1.4]])

    assert norm.vmin == pytest.approx(0.5)
    assert norm.vcenter == pytest.approx(1.0)
    assert norm.vmax == pytest.approx(2.0)


def test_token_count_colormap_uses_soft_blue_green_scale() -> None:
    cmap = build_token_count_colormap()

    assert cmap.name == "token_count_soft_blue_green"


def test_bar_rows_sort_by_lower_average_model_with_language_order_tie_break() -> None:
    rows = [
        {"language_code": "en", "model/a": "1.0", "model/b": "1.0"},
        {"language_code": "ja", "model/a": "1.4", "model/b": "1.1"},
        {"language_code": "zh", "model/a": "1.2", "model/b": "1.9"},
        {"language_code": "ko", "model/a": "1.2", "model/b": "1.2"},
    ]

    sorted_rows = sort_bar_rows(rows, ["model/a", "model/b"])

    assert [row["language_code"] for row in sorted_rows] == ["en", "zh", "ko", "ja"]
