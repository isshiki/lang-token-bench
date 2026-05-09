# Agent Instructions

## Project

- Project name: `lang-token-bench`
- Display name: Language Token Efficiency Benchmark
- Project purpose: A multilingual token efficiency benchmark across major LLMs.

## Communication Rule

ユーザーへの説明、進捗報告、確認事項は日本語で行う。

Project-facing files may remain in English. This includes README files, code,
CLI messages, comments, and commit messages.

## Safety Rules

- Do not read, inspect, print, create, modify, or commit any `.env` file.
- Only `.env.example` may be created or edited.
- The application may load `.env` at runtime through its normal CLI startup.
  Agents and contributors must not directly inspect or print the file.
- `.env` is for user-managed local secrets only and must stay ignored by Git.
- Never log API keys or environment variable values.
- The only API key currently required by this project is `OPENROUTER_API_KEY`.
- `.env.example` should contain only `OPENROUTER_API_KEY` unless the current
  workflow changes.
- Direct Anthropic, Gemini, and Hugging Face credentials are not part of the
  current workflow and should not be requested from users.
- Do not run API-backed benchmarks unless the user explicitly asks.
- Do not run standalone OpenRouter credit checks unless the user explicitly asks.
- For the OpenRouter usage backend, prefer `--dry-run`, `--limit 1`, `--model-id`, `--text-id`, and `--language-code`, and require `--yes` for real API calls.
- Real `openrouter-usage` runs must always perform before/after OpenRouter credit checks.
- OpenRouter usage requests default to `--max-output-tokens 16`; keep this
  explicit in dry-runs because some providers reject lower completion budgets.
- OpenRouter provider routing may be used to diagnose provider-specific
  failures. Prefer dry-runs first, then smallest possible `--yes` runs. For
  example, use `--provider-only anthropic --no-provider-fallbacks` or
  `--provider-ignore amazon-bedrock` when investigating Bedrock routing issues.

## Development Commands

```powershell
uv sync
uv sync --extra tiktoken
uv sync --extra openrouter
uv sync --extra viz
uv sync --extra openrouter --extra viz
uv run lang-token-bench run --counter simple
uv run lang-token-bench run --counter openai-tiktoken
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id openai/gpt-4o-mini --text-id short_instruction --limit 1
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id anthropic/claude-opus-4.7 --text-id short_instruction --limit 1 --provider-only anthropic --no-provider-fallbacks
uv run lang-token-bench run-suite --suite public_comparison_2026_04 --dry-run
uv run lang-token-bench openrouter credits
uv run lang-token-bench openrouter validate-models --suite main_2026_04
uv run lang-token-bench summarize --suite public_comparison_2026_04
uv run lang-token-bench summarize --suite all_2026_04 --debug-sources
uv run lang-token-bench plot --suite public_comparison_2026_04
uv run pytest
```

If pytest fails with `_overlapped` / `WinError 10106` in Codex/Windows environments, use:

```powershell
uv run pytest -p no:debugging
```

## Architecture Notes

- Core benchmark logic should remain independent from individual token counters.
- New backends should be added under `src/lang_token_bench/counters/`.
- Keep `counting_method` explicit.
- The current comparison strategy is OpenRouter observed usage, including
  OpenRouter-hosted Anthropic, Gemini, Qwen, Kimi, and related model IDs.
- OpenRouter observed usage and official tokenizer-style counts must remain clearly separated.
- OpenRouter credit balances must not be written to `results.csv` or `results.md`;
  use `run_summary.json` and `run_history.csv` for OpenRouter run accounting.
- Real OpenRouter usage runs should also write run-scoped artifacts under
  `outputs/runs/<run_id>/`.
- `run-suite` executes suite model IDs in order and writes a suite summary under
  `outputs/suite_runs/<suite_run_id>/suite_summary.json`.
- `run-suite` skips models with complete saved run-scoped results by default;
  use `--force` only when a deliberate re-run is needed.
- `run-suite --dry-run` must not call usage APIs or credit APIs.
- `run-suite` with OpenRouter-backed models requires `--yes` before real calls.
- `run-suite --continue-on-error` may proceed after a model failure, but must
  record `models_failed` and `failure_reasons` in the suite summary.
- OpenRouter Chat Completions errors may include safe response details, but must
  never print API keys, Authorization headers, request headers, or full payloads.
- OpenRouter usage benchmarks should keep completion output small while using
  `max_tokens: 16` by default for provider compatibility.
- Provider routing options map to OpenRouter's request-level `provider` object:
  `--provider-only`, `--provider-ignore`, `--provider-order`, and
  `--no-provider-fallbacks`. Routing config is safe to print and store, but API
  keys and headers are not.
- Benchmark suites are configured in `configs/benchmark_suites.yaml`.
- Summary generation reads saved result CSV files and filters by suite model IDs.
- Use `summarize --debug-sources` when validating which run folder supplied a
  model's summary rows.
- Summary generation writes latest top-level files and suite-scoped files under
  `outputs/summaries/<suite_name>/` using a safe folder name.
- Summary ratio CSV/Markdown include an `Avg` column and final `Avg` row;
  average rows exclude English. Heatmap CSV is long format with `is_average`.
- Summary generation also writes token count CSV/Markdown and token count
  heatmap CSV files using average observed input prompt token counts.
- Plot generation reads suite-scoped summary CSV files and writes figures under
  `outputs/summaries/<suite_name>/figures/`; it requires `uv sync --extra viz`.
- Plot generation should include both ratio and input-token-count heatmaps.
- Suite chart definitions live in `configs/benchmark_suites.yaml`.
- Plot labels should come from `configs/models.yaml` `short_name` and
  `configs/languages.yaml` `plot_label` with sensible fallbacks.
- Heatmaps should use the configured ratio scale centered on `1.0`; two-model
  bar charts should sort languages by `by_lower_average_model` unless configured
  otherwise.
- OpenRouter model validation may call `/api/v1/models`, but must not run Chat Completions.
- Reporters should remain under `src/lang_token_bench/reporters/`.
- Config-driven behavior should use `configs/models.yaml`, `configs/languages.yaml`, and `datasets/sample_texts.yaml`.

## Current Backend Status

- `simple`: implemented
- `openai-tiktoken`: implemented as optional tiktoken backend
- `openrouter-usage`: implemented OpenRouter observed usage counter
- `anthropic_api`: optional future reference backend stub, not used today
- `gemini_api`: optional future reference backend stub, not used today
- `hf_tokenizer`: optional future reference backend stub, not used today
- `playwright_web`: stub

## Testing Policy

- Run simple counter tests before and after backend changes.
- Do not require real API keys for default tests.
- API-backed tests should be skipped by default unless explicitly enabled.
