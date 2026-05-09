from __future__ import annotations

from lang_token_bench.config import (
    load_benchmark_suite,
    load_benchmark_suites,
    load_languages,
    load_models,
    load_sample_texts,
)


def test_default_config_loads_expected_languages_and_models() -> None:
    languages = load_languages()
    models = load_models()

    assert len(languages) == 8
    assert all(language.enabled for language in languages)
    assert {language.code for language in languages} == {
        "en",
        "ja",
        "zh",
        "ko",
        "es",
        "fr",
        "ar",
        "hi",
    }
    assert any(model.counter == "simple" and model.enabled for model in models)
    assert any(model.counter == "openai-tiktoken" for model in models)
    assert any(
        model.id == "openai/gpt-4o-mini"
        and model.counter == "openrouter-usage"
        and model.short_name == "GPT-4o mini"
        and not model.enabled
        for model in models
    )
    assert any(
        model.id == "anthropic/claude-opus-4.6"
        and model.counter == "openrouter-usage"
        and model.short_name == "Claude Opus 4.6"
        and not model.enabled
        for model in models
    )
    assert any(
        model.id == "anthropic/claude-haiku-4.5"
        and model.counter == "openrouter-usage"
        and model.short_name == "Claude Haiku 4.5"
        and not model.enabled
        for model in models
    )
    assert next(language for language in languages if language.code == "ja").plot_label == "Japanese"


def test_sample_texts_support_multiple_record_shape() -> None:
    texts = load_sample_texts()

    assert len(texts) == 5
    assert texts[0].id == "short_instruction"
    assert "en" in texts[0].contents
    assert "ja" in texts[0].contents
    assert {text.id for text in texts} == {
        "short_instruction",
        "medium_coding_instruction",
        "long_news_summary_instruction",
        "long_system_prompt",
        "very_long_article",
    }
    assert all(
        set(text.contents) == {"en", "ja", "zh", "ko", "es", "fr", "ar", "hi"}
        for text in texts
    )


def test_benchmark_suites_load_expected_model_groups() -> None:
    suites = load_benchmark_suites()

    assert {suite.name for suite in suites} >= {
        "main_2026_04",
        "budget_2026_04",
        "public_comparison_2026_04",
        "all_2026_04",
        "anthropic_comparison_2026_05",
        "gpt_comparison_2026_05",
    }
    main = load_benchmark_suite("main_2026_04")
    budget = load_benchmark_suite("budget_2026_04")
    all_suite = load_benchmark_suite("all_2026_04")
    public = load_benchmark_suite("public_comparison_2026_04")
    anthropic = load_benchmark_suite("anthropic_comparison_2026_05")
    gpt = load_benchmark_suite("gpt_comparison_2026_05")
    assert "openai/gpt-4o-mini" in budget.model_ids
    assert budget.model_ids
    expected_all = list(dict.fromkeys([*main.model_ids, *budget.model_ids]))
    assert all_suite.model_ids == expected_all
    assert public.charts
    assert public.charts[0].id == "openai_vs_anthropic_bar"
    assert public.charts[0].sort_languages == "by_lower_average_model"
    assert public.charts[0].show_value_labels is True
    assert public.charts[0].legend_position == "top"
    assert anthropic.model_ids == [
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-opus-4.7",
    ]
    assert anthropic.charts[0].id == "anthropic_opus46_vs_opus47_bar"
    assert gpt.model_ids == [
        "openai/gpt-5.4",
        "openai/gpt-5.5",
        "openai/gpt-4o-mini",
        "openai/gpt-oss-120b",
    ]
    assert gpt.charts[0].id == "gpt54_vs_gpt55_bar"
    assert gpt.charts[1].id == "gpt4o_mini_vs_gpt_oss120b_bar"
