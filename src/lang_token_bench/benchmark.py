from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lang_token_bench.config import (
    DEFAULT_LANGUAGES_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_SAMPLE_TEXTS_PATH,
    load_languages,
    load_models,
    load_sample_texts,
)
from lang_token_bench.counters import create_counter
from lang_token_bench.counters.openrouter_usage import DEFAULT_MAX_OUTPUT_TOKENS
from lang_token_bench.pricing import estimate_input_cost_usd
from lang_token_bench.schema import (
    BenchmarkPlanItem,
    BenchmarkResult,
    LanguageConfig,
    ModelConfig,
    SampleText,
)


def run_benchmark(
    *,
    languages_path: Path = DEFAULT_LANGUAGES_PATH,
    models_path: Path = DEFAULT_MODELS_PATH,
    sample_texts_path: Path = DEFAULT_SAMPLE_TEXTS_PATH,
    counter_filter: str | None = None,
    model_id_filter: str | None = None,
    text_id_filter: str | None = None,
    limit: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> list[BenchmarkResult]:
    plan = build_benchmark_plan(
        languages_path=languages_path,
        models_path=models_path,
        sample_texts_path=sample_texts_path,
        counter_filter=counter_filter,
        model_id_filter=model_id_filter,
        text_id_filter=text_id_filter,
        limit=limit,
    )
    return run_benchmark_plan(plan, max_output_tokens=max_output_tokens)


def run_benchmark_plan(
    plan: list[BenchmarkPlanItem],
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> list[BenchmarkResult]:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    counters = {
        item.model.counter: create_counter(
            item.model.counter,
            max_output_tokens=max_output_tokens,
        )
        for item in plan
    }

    count_results: list[tuple[BenchmarkPlanItem, int, str]] = []
    english_counts: dict[tuple[str, str], int] = {}

    for item in plan:
        count_result = counters[item.model.counter].count(
            item.sample_text.contents[item.language.code],
            item.model,
        )
        count_results.append((item, count_result.token_count, count_result.counting_method))
        if item.language.code == "en":
            english_counts[(item.model.id, item.sample_text.id)] = count_result.token_count

    results: list[BenchmarkResult] = []
    for item, token_count, counting_method in count_results:
        english_count = english_counts.get((item.model.id, item.sample_text.id))
        results.append(
            BenchmarkResult(
                model_id=item.model.id,
                provider=item.model.provider,
                counter=item.model.counter,
                counting_method=counting_method,
                language_code=item.language.code,
                language_name=item.language.name,
                text_id=item.sample_text.id,
                token_count=token_count,
                ratio_to_english=_ratio_to_english(token_count, english_count),
                input_price_per_1m_tokens=item.model.input_price_per_1m_tokens,
                estimated_input_cost_usd=estimate_input_cost_usd(
                    token_count,
                    item.model.input_price_per_1m_tokens,
                ),
                timestamp_utc=timestamp,
            )
        )

    return results


def build_benchmark_plan(
    *,
    languages_path: Path = DEFAULT_LANGUAGES_PATH,
    models_path: Path = DEFAULT_MODELS_PATH,
    sample_texts_path: Path = DEFAULT_SAMPLE_TEXTS_PATH,
    counter_filter: str | None = None,
    model_id_filter: str | None = None,
    text_id_filter: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkPlanItem]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer.")

    languages = [language for language in load_languages(languages_path) if language.enabled]
    models = _select_models(load_models(models_path), counter_filter)
    models = _filter_models_by_id(models, model_id_filter)
    sample_texts = _filter_sample_texts_by_id(load_sample_texts(sample_texts_path), text_id_filter)

    if not languages:
        raise ValueError("No enabled languages found.")
    if not models:
        raise ValueError("No models selected for benchmark.")
    if not sample_texts:
        raise ValueError("No sample texts found.")

    plan: list[BenchmarkPlanItem] = []
    for model in models:
        for sample_text in sample_texts:
            _validate_sample_contents(sample_text, languages)
            for language in languages:
                plan.append(
                    BenchmarkPlanItem(
                        model=model,
                        sample_text=sample_text,
                        language=language,
                    )
                )

    if limit is not None:
        plan = plan[:limit]
    if not plan:
        raise ValueError("Benchmark plan is empty.")
    return plan


def group_results_by_model_text(
    results: list[BenchmarkResult],
) -> dict[tuple[str, str], list[BenchmarkResult]]:
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[(result.model_id, result.text_id)].append(result)
    return dict(grouped)


def _select_models(
    models: list[ModelConfig],
    counter_filter: str | None,
) -> list[ModelConfig]:
    if counter_filter is None:
        return [model for model in models if model.enabled]

    normalized = counter_filter.strip().lower()
    selected = [model for model in models if model.counter.strip().lower() == normalized]
    if not selected:
        available = ", ".join(sorted({model.counter for model in models}))
        raise ValueError(
            f"No models found for counter '{counter_filter}'. Available configured counters: {available}"
        )
    return selected


def _filter_models_by_id(
    models: list[ModelConfig],
    model_id_filter: str | None,
) -> list[ModelConfig]:
    if model_id_filter is None:
        return models
    selected = [model for model in models if model.id == model_id_filter]
    if not selected:
        available = ", ".join(sorted({model.id for model in models}))
        raise ValueError(
            f"No models found for model id '{model_id_filter}'. Available selected model ids: {available}"
        )
    return selected


def _filter_sample_texts_by_id(
    sample_texts: list[SampleText],
    text_id_filter: str | None,
) -> list[SampleText]:
    if text_id_filter is None:
        return sample_texts
    selected = [sample_text for sample_text in sample_texts if sample_text.id == text_id_filter]
    if not selected:
        available = ", ".join(sorted({sample_text.id for sample_text in sample_texts}))
        raise ValueError(
            f"No sample texts found for text id '{text_id_filter}'. Available text ids: {available}"
        )
    return selected


def _validate_sample_contents(
    sample_text: SampleText,
    languages: list[LanguageConfig],
) -> None:
    missing = [language.code for language in languages if language.code not in sample_text.contents]
    if missing:
        raise ValueError(
            f"Sample text '{sample_text.id}' is missing enabled language contents: {', '.join(missing)}"
        )


def _ratio_to_english(token_count: int, english_count: int | None) -> float | None:
    if not english_count:
        return None
    return round(token_count / english_count, 6)
