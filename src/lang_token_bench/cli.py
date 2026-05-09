from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from lang_token_bench.benchmark import build_benchmark_plan, run_benchmark_plan
from lang_token_bench.config import (
    DEFAULT_BENCHMARK_SUITES_PATH,
    DEFAULT_LANGUAGES_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_TEXTS_PATH,
    load_benchmark_suite,
    load_languages,
    load_models,
    load_sample_texts,
)
from lang_token_bench.counters.base import CounterRequestError, CounterUnavailableError
from lang_token_bench.counters.openrouter_usage import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    OpenRouterProviderRouting,
)
from lang_token_bench.env import load_project_env
from lang_token_bench.openrouter_credits import (
    OpenRouterCreditsClient,
    format_credit_amount,
)
from lang_token_bench.openrouter_models import OpenRouterModelsClient
from lang_token_bench.plotting import plot_suite_figures
from lang_token_bench.reporters.csv_reporter import write_csv_report
from lang_token_bench.reporters.markdown_reporter import write_markdown_report
from lang_token_bench.run_tracking import (
    OpenRouterCreditRunSummary,
    SuiteRunSummary,
    append_run_history,
    build_run_id,
    build_suite_run_id,
    write_run_summary,
    write_suite_summary,
)
from lang_token_bench.schema import BenchmarkPlanItem, ModelConfig
from lang_token_bench.summary import (
    get_summary_source_info,
    load_results_csv,
    summarize_suite_results,
    write_summary_reports,
)


API_BACKED_COUNTERS = {
    "openrouter-usage",
    "openrouter_usage",
    "anthropic-api",
    "anthropic_api",
    "gemini-api",
    "gemini_api",
}
OPENROUTER_USAGE_COUNTERS = {
    "openrouter-usage",
    "openrouter_usage",
}


@dataclass(frozen=True)
class BenchmarkRunExecution:
    run_id: str
    rows_executed: int
    credit_summary: OpenRouterCreditRunSummary | None


@dataclass(frozen=True)
class SuiteModelPlan:
    model_id: str
    plan: list[BenchmarkPlanItem]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lang-token-bench",
        description="Language Token Efficiency Benchmark CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the token efficiency benchmark")
    run_parser.set_defaults(handler=_handle_run_command)
    run_parser.add_argument(
        "--counter",
        help="Run only models configured for this counter, for example: simple or openai-tiktoken",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results.csv and results.md",
    )
    run_parser.add_argument(
        "--languages",
        type=Path,
        default=DEFAULT_LANGUAGES_PATH,
        help="Path to languages.yaml",
    )
    run_parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_MODELS_PATH,
        help="Path to models.yaml",
    )
    run_parser.add_argument(
        "--texts",
        type=Path,
        default=DEFAULT_SAMPLE_TEXTS_PATH,
        help="Path to sample_texts.yaml",
    )
    run_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm real API requests for API-backed counters",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selected benchmark rows without counting tokens or calling APIs",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of benchmark rows/API requests",
    )
    run_parser.add_argument(
        "--model-id",
        help="Run only a specific model id from models.yaml",
    )
    run_parser.add_argument(
        "--text-id",
        help="Run only a specific sample text id from sample_texts.yaml",
    )
    run_parser.add_argument(
        "--language-code",
        action="append",
        metavar="CODES",
        help=(
            "Run only specific language codes, comma-separated or repeated. "
            "Example: --language-code en,hi"
        ),
    )
    run_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Completion token budget for OpenRouter usage requests",
    )
    _add_openrouter_provider_routing_args(run_parser)

    run_suite_parser = subparsers.add_parser(
        "run-suite",
        help="Run all models from a benchmark suite in order",
    )
    run_suite_parser.set_defaults(handler=_handle_run_suite_command)
    run_suite_parser.add_argument(
        "--suite",
        required=True,
        help="Suite name from configs/benchmark_suites.yaml",
    )
    run_suite_parser.add_argument(
        "--suites",
        type=Path,
        default=DEFAULT_BENCHMARK_SUITES_PATH,
        help="Path to benchmark_suites.yaml",
    )
    run_suite_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for suite and run outputs",
    )
    run_suite_parser.add_argument(
        "--languages",
        type=Path,
        default=DEFAULT_LANGUAGES_PATH,
        help="Path to languages.yaml",
    )
    run_suite_parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_MODELS_PATH,
        help="Path to models.yaml",
    )
    run_suite_parser.add_argument(
        "--texts",
        type=Path,
        default=DEFAULT_SAMPLE_TEXTS_PATH,
        help="Path to sample_texts.yaml",
    )
    run_suite_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm real API requests for API-backed suite models",
    )
    run_suite_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selected suite rows without counting tokens or calling APIs",
    )
    run_suite_parser.add_argument(
        "--limit",
        type=int,
        help="Limit benchmark rows per suite model",
    )
    run_suite_parser.add_argument(
        "--model-id",
        help="Run only a specific model id from the selected suite",
    )
    run_suite_parser.add_argument(
        "--text-id",
        help="Run only a specific sample text id from sample_texts.yaml",
    )
    run_suite_parser.add_argument(
        "--language-code",
        action="append",
        metavar="CODES",
        help=(
            "Run only specific language codes, comma-separated or repeated. "
            "Example: --language-code en,hi"
        ),
    )
    run_suite_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run suite models even when complete saved run results already exist",
    )
    run_suite_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later suite models if one model fails",
    )
    run_suite_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Completion token budget for OpenRouter usage requests",
    )
    _add_openrouter_provider_routing_args(run_suite_parser)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize saved benchmark results for a benchmark suite",
    )
    summarize_parser.set_defaults(handler=_handle_summarize_command)
    summarize_parser.add_argument(
        "--suite",
        required=True,
        help="Suite name from configs/benchmark_suites.yaml",
    )
    summarize_parser.add_argument(
        "--suites",
        type=Path,
        default=DEFAULT_BENCHMARK_SUITES_PATH,
        help="Path to benchmark_suites.yaml",
    )
    summarize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing saved runs and summary outputs",
    )
    summarize_parser.add_argument(
        "--languages",
        type=Path,
        default=DEFAULT_LANGUAGES_PATH,
        help="Path to languages.yaml",
    )
    summarize_parser.add_argument(
        "--debug-sources",
        action="store_true",
        help="Print source run and results.csv details used for each suite model",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate figures from a suite summary CSV",
    )
    plot_parser.set_defaults(handler=_handle_plot_command)
    plot_parser.add_argument(
        "--suite",
        required=True,
        help="Suite name from configs/benchmark_suites.yaml",
    )
    plot_parser.add_argument(
        "--suites",
        type=Path,
        default=DEFAULT_BENCHMARK_SUITES_PATH,
        help="Path to benchmark_suites.yaml",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing suite summary outputs",
    )
    plot_parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_MODELS_PATH,
        help="Path to models.yaml",
    )
    plot_parser.add_argument(
        "--languages",
        type=Path,
        default=DEFAULT_LANGUAGES_PATH,
        help="Path to languages.yaml",
    )

    openrouter_parser = subparsers.add_parser(
        "openrouter",
        help="OpenRouter utility commands",
    )
    openrouter_subparsers = openrouter_parser.add_subparsers(dest="openrouter_command")
    credits_parser = openrouter_subparsers.add_parser(
        "credits",
        help="Show OpenRouter credit usage and remaining credits",
    )
    credits_parser.set_defaults(handler=_handle_openrouter_credits_command)
    validate_parser = openrouter_subparsers.add_parser(
        "validate-models",
        help="Validate suite model ids against the OpenRouter models API",
    )
    validate_parser.set_defaults(handler=_handle_openrouter_validate_models_command)
    validate_parser.add_argument(
        "--suite",
        required=True,
        help="Suite name from configs/benchmark_suites.yaml",
    )
    validate_parser.add_argument(
        "--suites",
        type=Path,
        default=DEFAULT_BENCHMARK_SUITES_PATH,
        help="Path to benchmark_suites.yaml",
    )
    return parser


def _add_openrouter_provider_routing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider-only",
        action="append",
        metavar="SLUGS",
        help=(
            "OpenRouter provider slugs to allow, comma-separated or repeated. "
            "Example: --provider-only anthropic"
        ),
    )
    parser.add_argument(
        "--provider-ignore",
        action="append",
        metavar="SLUGS",
        help=(
            "OpenRouter provider slugs to ignore, comma-separated or repeated. "
            "Example: --provider-ignore amazon-bedrock"
        ),
    )
    parser.add_argument(
        "--provider-order",
        action="append",
        metavar="SLUGS",
        help=(
            "OpenRouter provider slugs to prioritize, comma-separated or repeated. "
            "Example: --provider-order anthropic,amazon-bedrock"
        ),
    )
    parser.add_argument(
        "--no-provider-fallbacks",
        action="store_true",
        help="Set OpenRouter provider.allow_fallbacks=false for the request",
    )


def main(argv: list[str] | None = None, *, load_env: bool = True) -> int:
    if load_env:
        load_project_env()

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except (CounterRequestError, CounterUnavailableError, NotImplementedError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _handle_run_command(args: argparse.Namespace) -> int:
    _validate_max_output_tokens(args.max_output_tokens)
    provider_routing = _build_openrouter_provider_routing(args)
    plan = build_benchmark_plan(
        languages_path=args.languages,
        models_path=args.models,
        sample_texts_path=args.texts,
        counter_filter=args.counter,
        model_id_filter=args.model_id,
        text_id_filter=args.text_id,
        language_code_filter=_parse_language_codes(args.language_code),
        limit=args.limit,
    )
    _validate_provider_routing_scope(plan, provider_routing)
    api_backed_counter = _find_api_backed_counter(plan)
    if args.dry_run:
        _print_dry_run(
            plan,
            args.limit,
            args.max_output_tokens,
            provider_routing,
        )
        return 0
    if api_backed_counter and not args.yes:
        raise ValueError(
            f"{api_backed_counter} requires --yes to run real API requests. "
            "Use --dry-run to preview."
        )

    _execute_benchmark_plan(
        plan,
        output_dir=args.output_dir,
        max_output_tokens=args.max_output_tokens,
        provider_routing=provider_routing,
    )
    return 0


def _handle_run_suite_command(args: argparse.Namespace) -> int:
    _validate_max_output_tokens(args.max_output_tokens)
    provider_routing = _build_openrouter_provider_routing(args)
    suite = load_benchmark_suite(args.suite, args.suites)
    model_ids = _select_suite_model_ids(suite.model_ids, args.model_id)
    model_plans = [
        SuiteModelPlan(
            model_id=model_id,
            plan=_build_suite_model_plan(
                model_id=model_id,
                languages_path=args.languages,
                models_path=args.models,
                sample_texts_path=args.texts,
                text_id_filter=args.text_id,
                language_code_filter=_parse_language_codes(args.language_code),
                limit=args.limit,
            ),
        )
        for model_id in model_ids
    ]
    run_model_plans, skipped_model_ids = _select_run_suite_models(
        model_plans,
        output_dir=args.output_dir,
        force=args.force,
    )
    selected_plan = [item for model_plan in model_plans for item in model_plan.plan]
    combined_plan = [item for model_plan in run_model_plans for item in model_plan.plan]
    _validate_provider_routing_scope(selected_plan, provider_routing)

    if args.dry_run:
        _print_suite_dry_run(
            suite_name=suite.name,
            selected_model_plans=model_plans,
            run_model_plans=run_model_plans,
            skipped_model_ids=skipped_model_ids,
            limit=args.limit,
            force=args.force,
            max_output_tokens=args.max_output_tokens,
            provider_routing=provider_routing,
        )
        return 0

    if args.force:
        print("Force mode: existing saved run results will not be skipped.")
    elif skipped_model_ids:
        print("Skipping suite models with complete saved run results:")
        for skipped_model_id in skipped_model_ids:
            print(f"- {skipped_model_id}")

    api_backed_counter = _find_api_backed_counter(combined_plan)
    if api_backed_counter and not args.yes:
        raise ValueError(
            f"run-suite with {api_backed_counter} requires --yes to run real API requests. "
            "Use --dry-run to preview."
        )

    suite_started_at = _utc_now()
    suite_run_id = build_suite_run_id(suite_started_at, suite.name)
    models_completed: list[str] = []
    models_failed: list[str] = []
    failure_reasons: dict[str, str] = {}
    run_ids: list[str] = []
    total_rows_executed = 0
    credit_summaries: list[OpenRouterCreditRunSummary] = []

    if not run_model_plans:
        print("No suite models need to run; complete saved run results already exist.")
        suite_summary_path = _write_suite_run_summary(
            suite_run_id=suite_run_id,
            suite_name=suite.name,
            started_at_utc=suite_started_at,
            output_dir=args.output_dir,
            models_requested=model_ids,
            models_completed=models_completed,
            models_skipped=skipped_model_ids,
            models_failed=models_failed,
            failure_reasons=failure_reasons,
            total_rows_executed=total_rows_executed,
            run_ids=run_ids,
            credit_summaries=credit_summaries,
        )
        print(f"Suite summary: {suite_summary_path}")
        return 0

    for model_plan in run_model_plans:
        print(f"Running suite model: {model_plan.model_id}")
        try:
            execution = _execute_benchmark_plan(
                model_plan.plan,
                output_dir=args.output_dir,
                max_output_tokens=args.max_output_tokens,
                provider_routing=provider_routing,
            )
            models_completed.append(model_plan.model_id)
            run_ids.append(execution.run_id)
            total_rows_executed += execution.rows_executed
            if execution.credit_summary is not None:
                credit_summaries.append(execution.credit_summary)
        except (CounterRequestError, CounterUnavailableError, NotImplementedError, ValueError) as exc:
            models_failed.append(model_plan.model_id)
            failure_reasons[model_plan.model_id] = str(exc)
            print(f"Suite model failed: {model_plan.model_id}", file=sys.stderr)
            print(f"Reason: {exc}", file=sys.stderr)
            if args.continue_on_error:
                print("Continuing because --continue-on-error was specified.", file=sys.stderr)
                continue
            _write_suite_run_summary(
                suite_run_id=suite_run_id,
                suite_name=suite.name,
                started_at_utc=suite_started_at,
                output_dir=args.output_dir,
                models_requested=model_ids,
                models_completed=models_completed,
                models_skipped=skipped_model_ids,
                models_failed=models_failed,
                failure_reasons=failure_reasons,
                total_rows_executed=total_rows_executed,
                run_ids=run_ids,
                credit_summaries=credit_summaries,
            )
            raise CounterRequestError(
                f"Suite model '{model_plan.model_id}' failed: {exc}"
            ) from exc

    suite_summary_path = _write_suite_run_summary(
        suite_run_id=suite_run_id,
        suite_name=suite.name,
        started_at_utc=suite_started_at,
        output_dir=args.output_dir,
        models_requested=model_ids,
        models_completed=models_completed,
        models_skipped=skipped_model_ids,
        models_failed=models_failed,
        failure_reasons=failure_reasons,
        total_rows_executed=total_rows_executed,
        run_ids=run_ids,
        credit_summaries=credit_summaries,
    )
    print(f"Suite summary: {suite_summary_path}")
    return 0


def _select_suite_model_ids(
    suite_model_ids: list[str],
    model_id_filter: str | None,
) -> list[str]:
    if model_id_filter is None:
        return suite_model_ids
    if model_id_filter not in suite_model_ids:
        available = ", ".join(suite_model_ids)
        raise ValueError(
            f"Model id '{model_id_filter}' is not in the selected suite. "
            f"Available suite model ids: {available}"
        )
    return [model_id_filter]


def _validate_max_output_tokens(max_output_tokens: int) -> None:
    if max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be a positive integer.")


def _build_openrouter_provider_routing(args: argparse.Namespace) -> OpenRouterProviderRouting:
    routing = OpenRouterProviderRouting(
        only=_parse_provider_slugs(getattr(args, "provider_only", None)),
        ignore=_parse_provider_slugs(getattr(args, "provider_ignore", None)),
        order=_parse_provider_slugs(getattr(args, "provider_order", None)),
        allow_fallbacks=False if getattr(args, "no_provider_fallbacks", False) else None,
    )
    if routing.only and routing.ignore:
        raise ValueError("--provider-only and --provider-ignore cannot be used together.")
    return routing


def _parse_provider_slugs(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    slugs: list[str] = []
    for value in values:
        for slug in value.split(","):
            normalized = slug.strip()
            if normalized:
                slugs.append(normalized)
    if not slugs:
        raise ValueError("Provider routing options require at least one provider slug.")
    return tuple(_unique_preserve_order(slugs))


def _parse_language_codes(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    codes: list[str] = []
    for value in values:
        for code in value.split(","):
            normalized = code.strip()
            if normalized:
                codes.append(normalized)
    if not codes:
        raise ValueError("--language-code requires at least one language code.")
    return _unique_preserve_order(codes)


def _validate_provider_routing_scope(
    plan: list[BenchmarkPlanItem],
    provider_routing: OpenRouterProviderRouting,
) -> None:
    if provider_routing.is_empty():
        return
    if not _find_openrouter_usage_counter(plan):
        raise ValueError(
            "OpenRouter provider routing options can only be used with "
            "the openrouter-usage counter."
        )


def _build_suite_model_plan(
    *,
    model_id: str,
    languages_path: Path,
    models_path: Path,
    sample_texts_path: Path,
    text_id_filter: str | None,
    language_code_filter: list[str] | None,
    limit: int | None,
) -> list[BenchmarkPlanItem]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer.")

    model = _resolve_suite_model(model_id, models_path)
    languages = [language for language in load_languages(languages_path) if language.enabled]
    if language_code_filter is not None:
        by_code = {language.code: language for language in languages}
        missing = [code for code in language_code_filter if code not in by_code]
        if missing:
            available = ", ".join(language.code for language in languages)
            raise ValueError(
                f"No enabled languages found for language code(s): {', '.join(missing)}. "
                f"Available enabled language codes: {available}"
            )
        languages = [by_code[code] for code in language_code_filter]
    sample_texts = load_sample_texts(sample_texts_path)
    if text_id_filter is not None:
        sample_texts = [text for text in sample_texts if text.id == text_id_filter]
        if not sample_texts:
            raise ValueError(f"No sample texts found for text id '{text_id_filter}'.")
    if not languages:
        raise ValueError("No enabled languages found.")
    if not sample_texts:
        raise ValueError("No sample texts found.")

    plan: list[BenchmarkPlanItem] = []
    for sample_text in sample_texts:
        missing = [
            language.code
            for language in languages
            if language.code not in sample_text.contents
        ]
        if missing:
            raise ValueError(
                f"Sample text '{sample_text.id}' is missing enabled language contents: "
                f"{', '.join(missing)}"
            )
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


def _resolve_suite_model(model_id: str, models_path: Path) -> ModelConfig:
    for model in load_models(models_path):
        if model.id == model_id:
            return model
    return ModelConfig(
        id=model_id,
        provider="openrouter",
        display_name=f"{model_id} via OpenRouter",
        counter="openrouter-usage",
        tokenizer_name=None,
        input_price_per_1m_tokens=None,
        enabled=False,
    )


def _select_run_suite_models(
    model_plans: list[SuiteModelPlan],
    *,
    output_dir: Path,
    force: bool,
) -> tuple[list[SuiteModelPlan], list[str]]:
    if force:
        return model_plans, []

    run_model_plans: list[SuiteModelPlan] = []
    skipped_model_ids: list[str] = []
    saved_run_results = _load_saved_run_result_keys(output_dir)
    for model_plan in model_plans:
        required_keys = {
            (item.sample_text.id, item.language.code)
            for item in model_plan.plan
        }
        saved_keys = saved_run_results.get(model_plan.model_id, set())
        if required_keys and required_keys <= saved_keys:
            skipped_model_ids.append(model_plan.model_id)
            continue
        run_model_plans.append(model_plan)
    return run_model_plans, skipped_model_ids


def _load_saved_run_result_keys(output_dir: Path) -> dict[str, set[tuple[str, str]]]:
    result_keys_by_model: dict[str, set[tuple[str, str]]] = {}
    for result_path in sorted((output_dir / "runs").glob("*/results.csv")):
        for result in load_results_csv(result_path):
            result_keys_by_model.setdefault(result.model_id, set()).add(
                (result.text_id, result.language_code)
            )
    return result_keys_by_model


def _write_suite_run_summary(
    *,
    suite_run_id: str,
    suite_name: str,
    started_at_utc: str,
    output_dir: Path,
    models_requested: list[str],
    models_completed: list[str],
    models_skipped: list[str],
    models_failed: list[str],
    failure_reasons: dict[str, str],
    total_rows_executed: int,
    run_ids: list[str],
    credit_summaries: list[OpenRouterCreditRunSummary],
) -> Path:
    credits_before_remaining: Decimal | None = None
    credits_after_remaining: Decimal | None = None
    if credit_summaries:
        credits_before_remaining = credit_summaries[0].credits_before.remaining_credits
        credits_after_remaining = credit_summaries[-1].credits_after.remaining_credits

    summary = SuiteRunSummary(
        suite_run_id=suite_run_id,
        suite_name=suite_name,
        started_at_utc=started_at_utc,
        ended_at_utc=_utc_now(),
        models_requested=models_requested,
        models_completed=models_completed,
        models_skipped=models_skipped,
        models_failed=models_failed,
        failure_reasons=failure_reasons,
        total_rows_executed=total_rows_executed,
        credits_before_remaining=credits_before_remaining,
        credits_after_remaining=credits_after_remaining,
        run_ids=run_ids,
    )
    return write_suite_summary(
        summary,
        output_dir / "suite_runs" / suite_run_id / "suite_summary.json",
    )


def _execute_benchmark_plan(
    plan: list[BenchmarkPlanItem],
    *,
    output_dir: Path,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    provider_routing: OpenRouterProviderRouting | None = None,
) -> BenchmarkRunExecution:
    started_at_utc = _utc_now()
    run_model_ids = _unique_preserve_order(item.model.id for item in plan)
    run_id = build_run_id(started_at_utc, run_model_ids)
    credits_client = OpenRouterCreditsClient()
    credits_before = None
    run_dir = None
    openrouter_usage_counter = _find_openrouter_usage_counter(plan)
    if openrouter_usage_counter:
        credits_before = credits_client.fetch()
        _print_openrouter_credits_before(credits_before)
        run_dir = output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark_plan(
        plan,
        max_output_tokens=max_output_tokens,
        openrouter_provider_routing=provider_routing,
    )
    csv_path = write_csv_report(results, output_dir / "results.csv")
    md_path = write_markdown_report(results, output_dir / "results.md")
    run_csv_path = None
    run_md_path = None
    if run_dir is not None:
        run_csv_path = write_csv_report(results, run_dir / "results.csv")
        run_md_path = write_markdown_report(results, run_dir / "results.md")

    print(f"Wrote {len(results)} benchmark rows.")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    if run_csv_path is not None and run_md_path is not None:
        print(f"Run CSV: {run_csv_path}")
        print(f"Run Markdown: {run_md_path}")

    credit_summary = None
    if openrouter_usage_counter and credits_before is not None:
        credits_after = credits_client.fetch()
        credit_summary = OpenRouterCreditRunSummary(
            run_id=run_id,
            started_at_utc=started_at_utc,
            ended_at_utc=_utc_now(),
            counter=openrouter_usage_counter,
            model_id=_summarize_plan_models(plan),
            text_id=_summarize_plan_texts(plan),
            rows_executed=len(results),
            credits_before=credits_before,
            credits_after=credits_after,
            provider_routing=(
                provider_routing.to_payload()
                if provider_routing and not provider_routing.is_empty()
                else None
            ),
        )
        summary_path = write_run_summary(credit_summary, output_dir / "run_summary.json")
        run_summary_path = None
        if run_dir is not None:
            run_summary_path = write_run_summary(credit_summary, run_dir / "run_summary.json")
        history_path = append_run_history(credit_summary, output_dir / "run_history.csv")
        _print_openrouter_credit_summary(credit_summary)
        print(f"Run summary: {summary_path}")
        if run_summary_path is not None:
            print(f"Run-specific summary: {run_summary_path}")
        print(f"Run history: {history_path}")
    return BenchmarkRunExecution(
        run_id=run_id,
        rows_executed=len(results),
        credit_summary=credit_summary,
    )


def _handle_openrouter_credits_command(_args: argparse.Namespace) -> int:
    credits = OpenRouterCreditsClient().fetch()
    print("OpenRouter credits:")
    print(f"Total credits: {format_credit_amount(credits.total_credits)}")
    print(f"Total usage: {format_credit_amount(credits.total_usage)}")
    print(f"Remaining credits: {format_credit_amount(credits.remaining_credits)}")
    return 0


def _handle_summarize_command(args: argparse.Namespace) -> int:
    suite = load_benchmark_suite(args.suite, args.suites)
    summary = summarize_suite_results(
        suite=suite,
        output_dir=args.output_dir,
        languages_path=args.languages,
    )
    report_paths = write_summary_reports(summary, args.output_dir)
    if args.debug_sources:
        _print_summary_debug_sources(
            suite_model_ids=suite.model_ids,
            source_info=get_summary_source_info(
                suite=suite,
                output_dir=args.output_dir,
            ),
        )
    print(f"Summary suite: {suite.name}")
    print(f"Models: {', '.join(suite.model_ids)}")
    print(f"Suite CSV: {report_paths.suite_csv}")
    print(f"Suite Markdown: {report_paths.suite_markdown}")
    print(f"Suite heatmap CSV: {report_paths.suite_heatmap_csv}")
    print(f"Suite token count CSV: {report_paths.suite_token_count_csv}")
    print(f"Suite token count Markdown: {report_paths.suite_token_count_markdown}")
    print(f"Suite token count heatmap CSV: {report_paths.suite_token_count_heatmap_csv}")
    print(f"Suite relative token count CSV: {report_paths.suite_relative_token_count_csv}")
    print(f"Suite relative token count Markdown: {report_paths.suite_relative_token_count_markdown}")
    print(f"Suite relative token count heatmap CSV: {report_paths.suite_relative_token_count_heatmap_csv}")
    print(f"Suite weighted ratio CSV: {report_paths.suite_weighted_ratio_csv}")
    print(f"Suite weighted ratio Markdown: {report_paths.suite_weighted_ratio_markdown}")
    print(f"Suite weighted ratio heatmap CSV: {report_paths.suite_weighted_ratio_heatmap_csv}")
    print(f"Suite excess tokens CSV: {report_paths.suite_excess_tokens_csv}")
    print(f"Suite excess tokens Markdown: {report_paths.suite_excess_tokens_markdown}")
    print(f"Suite excess tokens heatmap CSV: {report_paths.suite_excess_tokens_heatmap_csv}")
    print(f"Latest CSV: {report_paths.latest_csv}")
    print(f"Latest Markdown: {report_paths.latest_markdown}")
    print(f"Latest heatmap CSV: {report_paths.latest_heatmap_csv}")
    print(f"Latest token count CSV: {report_paths.latest_token_count_csv}")
    print(f"Latest token count Markdown: {report_paths.latest_token_count_markdown}")
    print(f"Latest token count heatmap CSV: {report_paths.latest_token_count_heatmap_csv}")
    print(f"Latest relative token count CSV: {report_paths.latest_relative_token_count_csv}")
    print(f"Latest relative token count Markdown: {report_paths.latest_relative_token_count_markdown}")
    print(f"Latest relative token count heatmap CSV: {report_paths.latest_relative_token_count_heatmap_csv}")
    print(f"Latest weighted ratio CSV: {report_paths.latest_weighted_ratio_csv}")
    print(f"Latest weighted ratio Markdown: {report_paths.latest_weighted_ratio_markdown}")
    print(f"Latest weighted ratio heatmap CSV: {report_paths.latest_weighted_ratio_heatmap_csv}")
    print(f"Latest excess tokens CSV: {report_paths.latest_excess_tokens_csv}")
    print(f"Latest excess tokens Markdown: {report_paths.latest_excess_tokens_markdown}")
    print(f"Latest excess tokens heatmap CSV: {report_paths.latest_excess_tokens_heatmap_csv}")
    return 0


def _print_summary_debug_sources(*, suite_model_ids: list[str], source_info) -> None:
    info_by_model: dict[str, list] = {}
    for info in source_info:
        info_by_model.setdefault(info.model_id, []).append(info)

    print("Summary source debug:")
    for model_id in suite_model_ids:
        print(f"Model: {model_id}")
        model_sources = info_by_model.get(model_id, [])
        if not model_sources:
            print("- no selected saved rows")
            continue
        for info in model_sources:
            timestamp_range = _format_timestamp_range(
                info.timestamp_start_utc,
                info.timestamp_end_utc,
            )
            print(f"- run_id: {info.run_id}")
            print(f"  results_csv: {info.path}")
            print(f"  adopted_rows: {info.adopted_rows_count}")
            print(f"  source_rows: {info.source_rows_count}")
            print(f"  timestamp_range: {timestamp_range}")
            print(f"  source_model_ids: {', '.join(info.source_model_ids)}")


def _format_timestamp_range(start: str | None, end: str | None) -> str:
    if start is None and end is None:
        return "unknown"
    if start == end:
        return start or "unknown"
    return f"{start or 'unknown'} to {end or 'unknown'}"


def _handle_plot_command(args: argparse.Namespace) -> int:
    suite = load_benchmark_suite(args.suite, args.suites)
    outputs = plot_suite_figures(
        suite=suite,
        output_dir=args.output_dir,
        models_path=args.models,
        languages_path=args.languages,
    )
    print(f"Plot suite: {suite.name}")
    for output in outputs:
        print(f"{output.label} PNG: {output.png_path}")
        print(f"{output.label} SVG: {output.svg_path}")
    return 0


def _handle_openrouter_validate_models_command(args: argparse.Namespace) -> int:
    suite = load_benchmark_suite(args.suite, args.suites)
    available_model_ids = OpenRouterModelsClient().fetch_model_ids()
    missing = [
        model_id
        for model_id in suite.model_ids
        if model_id not in available_model_ids
    ]
    print(f"OpenRouter model validation for suite: {suite.name}")
    print(f"Models checked: {len(suite.model_ids)}")
    if missing:
        print("Missing model IDs:")
        for model_id in missing:
            print(f"- {model_id}")
        raise ValueError(
            f"{len(missing)} model id(s) from suite '{suite.name}' were not found in OpenRouter."
        )
    print("All suite model IDs were found in OpenRouter.")
    return 0


def _find_openrouter_usage_counter(plan: list[BenchmarkPlanItem]) -> str | None:
    for item in plan:
        counter = item.model.counter.strip().lower()
        if counter in OPENROUTER_USAGE_COUNTERS:
            return item.model.counter
    return None


def _find_api_backed_counter(plan: list[BenchmarkPlanItem]) -> str | None:
    for item in plan:
        counter = item.model.counter.strip().lower()
        if counter in API_BACKED_COUNTERS:
            return item.model.counter
    return None


def _print_openrouter_credits_before(credits) -> None:
    print("OpenRouter credits before:")
    print(f"Total credits: {format_credit_amount(credits.total_credits)}")
    print(f"Total usage: {format_credit_amount(credits.total_usage)}")
    print(f"Remaining credits: {format_credit_amount(credits.remaining_credits)}")


def _print_openrouter_credit_summary(summary: OpenRouterCreditRunSummary) -> None:
    print("OpenRouter credit summary:")
    print(f"Credits before: {format_credit_amount(summary.credits_before.remaining_credits)}")
    print(f"Credits after: {format_credit_amount(summary.credits_after.remaining_credits)}")
    print(f"Credits used: {format_credit_amount(summary.credits_used)}")
    print(f"Rows executed: {summary.rows_executed}")


def _print_dry_run(
    plan: list[BenchmarkPlanItem],
    limit: int | None,
    max_output_tokens: int | None = None,
    provider_routing: OpenRouterProviderRouting | None = None,
) -> None:
    models = _unique_preserve_order(item.model.id for item in plan)
    text_ids = _unique_preserve_order(item.sample_text.id for item in plan)
    languages = _unique_preserve_order(
        f"{item.language.code} ({item.language.name})"
        for item in plan
    )

    print("Dry run: no token counting or API requests were performed.")
    print(f"Planned benchmark rows: {len(plan)}")
    if limit is not None:
        print(f"Limit: {limit}")
    if max_output_tokens is not None and _find_openrouter_usage_counter(plan):
        print(f"Max output tokens: {max_output_tokens}")
    _print_openrouter_provider_routing(plan, provider_routing)
    print("Models:")
    for model_id in models:
        print(f"- {model_id}")
    print("Text IDs:")
    for text_id in text_ids:
        print(f"- {text_id}")
    print("Languages:")
    for language in languages:
        print(f"- {language}")


def _print_suite_dry_run(
    *,
    suite_name: str,
    selected_model_plans: list[SuiteModelPlan],
    run_model_plans: list[SuiteModelPlan],
    skipped_model_ids: list[str],
    limit: int | None,
    force: bool,
    max_output_tokens: int,
    provider_routing: OpenRouterProviderRouting | None,
) -> None:
    selected_plan = [item for model_plan in selected_model_plans for item in model_plan.plan]
    plan = [item for model_plan in run_model_plans for item in model_plan.plan]
    text_ids = _unique_preserve_order(item.sample_text.id for item in selected_plan)
    languages = _unique_preserve_order(
        f"{item.language.code} ({item.language.name})"
        for item in selected_plan
    )

    print("Suite dry run: no token counting, credit checks, or API requests were performed.")
    print(f"Suite: {suite_name}")
    print(f"Force: {str(force).lower()}")
    print(f"Max output tokens: {max_output_tokens}")
    _print_openrouter_provider_routing(selected_plan, provider_routing)
    print(f"Planned benchmark rows: {len(plan)}")
    if limit is not None:
        print(f"Limit: {limit}")
    print("Text IDs:")
    for text_id in text_ids:
        print(f"- {text_id}")
    print("Languages:")
    for language in languages:
        print(f"- {language}")
    print("Models to run:")
    if run_model_plans:
        for model_plan in run_model_plans:
            print(f"- {model_plan.model_id}")
    else:
        print("- none")
    print("Models to skip:")
    if skipped_model_ids:
        for model_id in skipped_model_ids:
            print(f"- {model_id}")
    else:
        print("- none")
    rows_by_model: dict[str, int] = {}
    for item in plan:
        rows_by_model[item.model.id] = rows_by_model.get(item.model.id, 0) + 1
    print("Rows by model:")
    if rows_by_model:
        for model_id, row_count in rows_by_model.items():
            print(f"- {model_id}: {row_count}")
    else:
        print("- none")


def _print_openrouter_provider_routing(
    plan: list[BenchmarkPlanItem],
    provider_routing: OpenRouterProviderRouting | None,
) -> None:
    if provider_routing is None or provider_routing.is_empty():
        return
    if not _find_openrouter_usage_counter(plan):
        return
    print("OpenRouter provider routing:")
    for key, value in provider_routing.to_payload().items():
        print(f"- {key}: {_format_provider_routing_value(value)}")


def _format_provider_routing_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _unique_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _summarize_plan_models(plan: list[BenchmarkPlanItem]) -> str:
    return ", ".join(_unique_preserve_order(item.model.id for item in plan))


def _summarize_plan_texts(plan: list[BenchmarkPlanItem]) -> str:
    return ", ".join(_unique_preserve_order(item.sample_text.id for item in plan))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
