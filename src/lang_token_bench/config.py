from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lang_token_bench.schema import (
    BenchmarkSuiteConfig,
    ChartConfig,
    LanguageConfig,
    ModelConfig,
    SampleText,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANGUAGES_PATH = PROJECT_ROOT / "configs" / "languages.yaml"
DEFAULT_MODELS_PATH = PROJECT_ROOT / "configs" / "models.yaml"
DEFAULT_BENCHMARK_SUITES_PATH = PROJECT_ROOT / "configs" / "benchmark_suites.yaml"
DEFAULT_SAMPLE_TEXTS_PATH = PROJECT_ROOT / "datasets" / "sample_texts.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at the top level: {path}")
    return data


def load_languages(path: Path = DEFAULT_LANGUAGES_PATH) -> list[LanguageConfig]:
    data = _load_yaml_mapping(path)
    raw_languages = data.get("languages")
    if not isinstance(raw_languages, list):
        raise ValueError(f"{path} must contain a 'languages' list")

    languages: list[LanguageConfig] = []
    for item in raw_languages:
        if not isinstance(item, dict):
            raise ValueError(f"Each language entry in {path} must be a mapping")
        languages.append(
            LanguageConfig(
                code=str(item["code"]),
                name=str(item["name"]),
                native_name=str(item["native_name"]),
                enabled=bool(item.get("enabled", True)),
                plot_label=(
                    None if item.get("plot_label") is None else str(item["plot_label"])
                ),
            )
        )
    return languages


def load_models(path: Path = DEFAULT_MODELS_PATH) -> list[ModelConfig]:
    data = _load_yaml_mapping(path)
    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        raise ValueError(f"{path} must contain a 'models' list")

    models: list[ModelConfig] = []
    for item in raw_models:
        if not isinstance(item, dict):
            raise ValueError(f"Each model entry in {path} must be a mapping")
        price = item.get("input_price_per_1m_tokens")
        models.append(
            ModelConfig(
                id=str(item["id"]),
                provider=str(item["provider"]),
                display_name=str(item["display_name"]),
                counter=str(item["counter"]),
                tokenizer_name=(
                    None if item.get("tokenizer_name") is None else str(item["tokenizer_name"])
                ),
                input_price_per_1m_tokens=None if price is None else float(price),
                enabled=bool(item.get("enabled", True)),
                short_name=None if item.get("short_name") is None else str(item["short_name"]),
            )
        )
    return models


def load_sample_texts(path: Path = DEFAULT_SAMPLE_TEXTS_PATH) -> list[SampleText]:
    data = _load_yaml_mapping(path)
    raw_texts = data.get("texts")
    if not isinstance(raw_texts, list):
        raise ValueError(f"{path} must contain a 'texts' list")

    texts: list[SampleText] = []
    for item in raw_texts:
        if not isinstance(item, dict):
            raise ValueError(f"Each text entry in {path} must be a mapping")
        contents = item.get("contents")
        if not isinstance(contents, dict):
            raise ValueError(f"Text entry {item.get('id', '<unknown>')} must contain contents")
        texts.append(
            SampleText(
                id=str(item["id"]),
                description=str(item.get("description", "")),
                contents={str(code): str(text) for code, text in contents.items()},
            )
        )
    return texts


def load_benchmark_suites(
    path: Path = DEFAULT_BENCHMARK_SUITES_PATH,
) -> list[BenchmarkSuiteConfig]:
    data = _load_yaml_mapping(path)
    raw_suites = data.get("suites")
    if not isinstance(raw_suites, list):
        raise ValueError(f"{path} must contain a 'suites' list")

    suites: list[BenchmarkSuiteConfig] = []
    for item in raw_suites:
        if not isinstance(item, dict):
            raise ValueError(f"Each suite entry in {path} must be a mapping")
        raw_model_ids = item.get("model_ids")
        if not isinstance(raw_model_ids, list) or not raw_model_ids:
            raise ValueError(f"Suite {item.get('name', '<unknown>')} must contain model_ids")
        suites.append(
            BenchmarkSuiteConfig(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                model_ids=[str(model_id) for model_id in raw_model_ids],
                charts=_load_suite_charts(item.get("charts"), item.get("name", "<unknown>")),
            )
        )
    return suites


def load_benchmark_suite(
    suite_name: str,
    path: Path = DEFAULT_BENCHMARK_SUITES_PATH,
) -> BenchmarkSuiteConfig:
    suites = load_benchmark_suites(path)
    for suite in suites:
        if suite.name == suite_name:
            return suite
    available = ", ".join(sorted(suite.name for suite in suites))
    raise ValueError(f"Unknown benchmark suite '{suite_name}'. Available suites: {available}")


def _load_suite_charts(raw_charts: Any, suite_name: Any) -> list[ChartConfig]:
    if raw_charts is None:
        return []
    if not isinstance(raw_charts, list):
        raise ValueError(f"Suite {suite_name} charts must be a list")

    charts: list[ChartConfig] = []
    for item in raw_charts:
        if not isinstance(item, dict):
            raise ValueError(f"Each chart entry in suite {suite_name} must be a mapping")
        raw_model_ids = item.get("model_ids")
        if not isinstance(raw_model_ids, list) or not raw_model_ids:
            raise ValueError(
                f"Chart {item.get('id', '<unknown>')} in suite {suite_name} must contain model_ids"
            )
        chart_id = str(item["id"])
        title = item.get("title", chart_id)
        charts.append(
            ChartConfig(
                id=chart_id,
                type=str(item["type"]),
                title=chart_id if title is None else str(title),
                model_ids=[str(model_id) for model_id in raw_model_ids],
                output_name=str(item.get("output_name", chart_id)),
                sort_languages=str(item.get("sort_languages", "by_lower_average_model")),
                show_value_labels=bool(item.get("show_value_labels", True)),
                legend_position=str(item.get("legend_position", "top")),
            )
        )
    return charts
