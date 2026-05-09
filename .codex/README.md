# Codex Notes

This folder contains Codex-specific notes and workflows for this repository.

The canonical agent instructions are in [`AGENTS.md`](../AGENTS.md).

## Windows/Codex Pytest Workaround

Some Codex Desktop or Windows environments may fail while importing
`_overlapped`, often surfaced as `WinError 10106` during pytest startup.
If that happens, run pytest without pytest's debugging plugin:

```powershell
uv run pytest -p no:debugging
```

## Network And API Operations

Network and API operations should be done cautiously and only when explicitly
requested by the user. Do not run API-backed benchmarks by default.

For OpenRouter usage checks, preview first:

```powershell
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id openai/gpt-4o-mini --text-id short_instruction --limit 1
```

Use `--language-code` when a smaller language subset is safer for paid or
unstable API runs. Include `en` when ratios for another language must be
computed in the same run.

OpenRouter usage requests default to `--max-output-tokens 16`. Keep this value
visible in dry-runs because some providers reject smaller completion budgets.

OpenRouter provider routing options can help diagnose provider-specific
failures. Use dry-runs first, then the smallest possible real request. For
example:

```powershell
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id anthropic/claude-opus-4.7 --text-id short_instruction --limit 1 --provider-only anthropic --no-provider-fallbacks
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id anthropic/claude-opus-4.7 --text-id short_instruction --limit 1 --provider-ignore amazon-bedrock
```

For suite runs, preview first:

```powershell
uv run lang-token-bench run-suite --suite public_comparison_2026_04 --dry-run
```

`run-suite --dry-run` must not call usage APIs or credit APIs. Real OpenRouter
suite runs require `--yes`.

`run-suite` skips models with complete saved run-scoped results by default to
avoid duplicate paid calls. Use `--force` only when the user explicitly wants a
re-run.

By default, a suite stops when one model fails. Use `--continue-on-error` only
when the user explicitly accepts partial completion; failed models and
`failure_reasons` must be written to the suite summary.

Real OpenRouter API calls require `OPENROUTER_API_KEY` in the process
environment or the application-loaded project `.env`, plus the explicit `--yes`
flag.

`OPENROUTER_API_KEY` is the only API key currently required by this project.
Anthropic, Gemini, Qwen, Kimi, and related models are compared through
OpenRouter model IDs. Direct Anthropic, Gemini, and Hugging Face credentials
are optional future reference-backend concerns and should not be requested for
the current workflow.

Real `openrouter-usage` benchmark runs automatically check OpenRouter credits
before and after the usage requests. If the pre-run credit check fails, the
benchmark must stop before sending usage requests.

OpenRouter credit checks are available with:

```powershell
uv run lang-token-bench openrouter credits
```

OpenRouter model ID validation is available with:

```powershell
uv run lang-token-bench openrouter validate-models --suite main_2026_04
```

This command may call `/api/v1/models`, but it must not run Chat Completions.

Credit balances are displayed in the CLI and, for OpenRouter benchmark runs,
written to `outputs/run_summary.json` and appended to `outputs/run_history.csv`.
They must not be written to `outputs/results.csv` or `outputs/results.md`.
Run-scoped OpenRouter artifacts are stored under `outputs/runs/<run_id>/`.

OpenRouter Chat Completions errors may display safe response details such as
`error.message`, `error.code`, `error.type`, and `error.metadata`. Never print
API keys, Authorization headers, request headers, or full request payloads.
Provider routing configuration is safe to display because it contains provider
slugs only; it should be recorded in run summaries when used.

Benchmark suites are configured in `configs/benchmark_suites.yaml`. Saved
results can be summarized with:

```powershell
uv run lang-token-bench summarize --suite public_comparison_2026_04
uv run lang-token-bench summarize --suite all_2026_04 --debug-sources
```

Figures can be generated from saved suite summaries after installing the
optional visualization dependency:

```powershell
uv sync --extra viz
uv run lang-token-bench plot --suite public_comparison_2026_04
```

Summary output files are written under `outputs/` and should be generated from
saved results, not by rerunning paid benchmark calls.
Latest summary files remain at the top level, and suite-scoped summary files
are written under `outputs/summaries/<suite_name>/` using a safe folder name.
Summary ratio CSV/Markdown include an `Avg` column and final `Avg` row. The
heatmap CSV is long format and uses `is_average` to mark average cells.
Plot outputs are written under `outputs/summaries/<suite_name>/figures/`.
Summaries also include token count CSV/Markdown outputs and a token count
heatmap for average observed input prompt tokens.
Summaries also include relative token count CSV/Markdown outputs. Relative
token count normalizes every token-count cell by the minimum token-count cell
in the table, so `1.00x` means the lowest observed language/model cell.
Summaries also include weighted ratio and excess token CSV/Markdown outputs.
Weighted ratio uses total language prompt tokens divided by total English
prompt tokens, while excess tokens use total language prompt tokens minus the
matching English total. Average rows should exclude English.
Plots should use model `short_name` and language `plot_label` from the config
files when present.
Heatmaps use a diverging ratio scale centered on `1.0`. Two-model bar charts
sort languages by the lower-average model's non-English ratios by default.

Suite run summaries are written to:

```text
outputs/suite_runs/<suite_run_id>/suite_summary.json
```

Never inspect `.env`. Do not read, print, edit, or commit `.env` files.
The application may load `.env` at runtime, but Codex agents must not inspect
its contents. Users can copy `.env.example` to `.env` and fill in
`OPENROUTER_API_KEY`; `.env.example` should contain only that key for the
current workflow. `.env` is ignored by Git and must never be committed.
