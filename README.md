# Language Token Efficiency Benchmark

**A multilingual token efficiency benchmark across major LLMs.**

日本語名: **言語別トークン効率ベンチマーク**  
日本語副題: **主要LLMにおける多言語トークン効率の比較**

## Overview

Language Token Efficiency Benchmark is a Python CLI tool for comparing token counts across languages for aligned multilingual text samples.

This project primarily benchmarks observed prompt token usage through
OpenRouter. Anthropic, Gemini, Qwen, Kimi, and other models are compared
through OpenRouter model IDs in the current workflow. Direct tokenizer and
direct provider backends are reference paths or future work.

This project measures:

- Input token count by language
- Ratio to English (`ratio_to_english`)
- Estimated input cost when model pricing is configured
- Differences between counting backends and counting methods

The informal phrase **Token Tax** is sometimes used to describe cases where one language requires more tokens than another for equivalent content. This project uses the neutral name **Language Token Efficiency Benchmark** because results depend heavily on translation quality, text length, expression density, terminology, and tokenizer behavior.

## Quick Start

Install the OpenRouter and visualization extras:

```powershell
uv sync --extra openrouter --extra viz
```

Create a local `.env` file:

```powershell
copy .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY`. Do not print, log, or commit the key.
The app loads `.env` at runtime when present, but contributors and agents must
not inspect or print `.env` contents.

Then run the public comparison workflow:

```powershell
uv run lang-token-bench openrouter validate-models --suite public_comparison_2026_04
uv run lang-token-bench run-suite --suite public_comparison_2026_04 --dry-run
uv run lang-token-bench run-suite --suite public_comparison_2026_04 --yes
uv run lang-token-bench summarize --suite public_comparison_2026_04
uv run lang-token-bench plot --suite public_comparison_2026_04
```

`--dry-run` does not call usage APIs or credit APIs. `--yes` is required before
real OpenRouter usage requests. Real `openrouter-usage` runs automatically
check OpenRouter credits before and after the benchmark.

## Typical Workflow

1. Install optional dependencies with `uv sync --extra openrouter --extra viz`.
2. Configure `OPENROUTER_API_KEY` in a local `.env` file or OS environment.
3. Validate model IDs with `openrouter validate-models`.
4. Preview planned rows with `run-suite --dry-run`.
5. Run a suite with `run-suite --yes`.
6. Summarize saved run results with `summarize --suite <suite_name>`.
7. Generate figures with `plot --suite <suite_name>`.
8. Check outputs under `outputs/summaries/<suite_name>/`.

## Suites

| Suite | Purpose |
| --- | --- |
| `public_comparison_2026_04` | Main public-facing comparison and figures |
| `anthropic_comparison_2026_05` | Anthropic-only comparison across Haiku, Sonnet, Opus 4.6, and Opus 4.7 |
| `all_2026_04` | All measured models, including extra comparisons |
| `budget_2026_04` | Low-cost model checks |
| `main_2026_04` | Major model candidates |

## Important Caveats

This tool compares token counts for same-intent multilingual text samples. Results are not absolute measurements of language quality or model capability.

Translation quality, sentence length, compression, style, technical terminology, and script all affect token counts. A small sample should be treated as a diagnostic baseline, not as a definitive ranking.

OpenRouter usage results must be treated separately from official tokenizer results. The OpenRouter backend compares observed `usage.prompt_tokens` through OpenRouter, not standalone tokenizer output.

## Install

This project uses `uv`.

```powershell
uv sync
```

Useful extras:

- `uv sync --extra openrouter`: OpenRouter observed usage backend.
- `uv sync --extra viz`: Matplotlib figure generation.
- `uv sync --extra tiktoken`: optional OpenAI tokenizer reference backend.

If `tiktoken` is not installed, the CLI will show:

```text
openai-tiktoken counter requires the optional dependency. Install it with: uv sync --extra tiktoken
```

## Usage

### Local Smoke Test

Run the simple local baseline counter:

```powershell
uv run lang-token-bench run --counter simple
```

Run the enabled default benchmark:

```powershell
uv run lang-token-bench run
```

Run the OpenAI tiktoken counter after installing the optional dependency:

```powershell
uv run lang-token-bench run --counter openai-tiktoken
```

Write output files to a specific directory:

```powershell
uv run lang-token-bench run --output-dir outputs
```

### OpenRouter Model Validation

Validate OpenRouter model IDs from a benchmark suite without running a
benchmark:

```powershell
uv run lang-token-bench openrouter validate-models --suite public_comparison_2026_04
```

Check OpenRouter credits before or after an API-backed benchmark:

```powershell
uv run lang-token-bench openrouter credits
```

### Single-Model OpenRouter Run

Preview a single OpenRouter usage run without sending API requests:

```powershell
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id openai/gpt-4o-mini --text-id short_instruction --limit 1
```

The OpenRouter completion budget defaults to `--max-output-tokens 16`:

```powershell
uv run lang-token-bench run --counter openrouter-usage --dry-run --model-id openai/gpt-4o-mini --text-id short_instruction --limit 1 --max-output-tokens 16
```

Run a real OpenRouter usage request only after setting `OPENROUTER_API_KEY`
through the process environment or a local `.env` file:

```powershell
uv run lang-token-bench run --counter openrouter-usage --model-id openai/gpt-4o-mini --text-id short_instruction --limit 1 --yes
```

To steer OpenRouter away from a provider or toward a specific provider, use
provider routing options. For example, to test Anthropic-hosted Claude only:

```powershell
uv run lang-token-bench run --counter openrouter-usage --model-id anthropic/claude-opus-4.7 --text-id short_instruction --limit 1 --provider-only anthropic --no-provider-fallbacks --yes
```

To avoid Amazon Bedrock while still allowing OpenRouter to route elsewhere:

```powershell
uv run lang-token-bench run --counter openrouter-usage --model-id anthropic/claude-opus-4.7 --text-id short_instruction --limit 1 --provider-ignore amazon-bedrock --yes
```

When `openrouter-usage` runs, the CLI always checks OpenRouter credits before
and after the benchmark. There is no skip option for this safety check.

### Suite Run

Preview a benchmark suite without sending API or credit requests:

```powershell
uv run lang-token-bench run-suite --suite public_comparison_2026_04 --dry-run
```

Run all models from a suite in order:

```powershell
uv run lang-token-bench run-suite --suite public_comparison_2026_04 --yes
```

For OpenRouter models, `run-suite` also requires `--yes` before real API calls.
Use `--model-id` to run one model from the suite and `--text-id` to run one
sample text. Use `--language-code` to run a smaller language subset:

```powershell
uv run lang-token-bench run-suite --suite budget_2026_04 --model-id openai/gpt-4o-mini --text-id short_instruction --yes
uv run lang-token-bench run-suite --suite anthropic_comparison_2026_05 --model-id anthropic/claude-opus-4.6 --text-id long_news_summary_instruction --language-code en,hi --provider-only anthropic --no-provider-fallbacks --yes
```

Provider routing options also work with `run-suite`:

```powershell
uv run lang-token-bench run-suite --suite anthropic_comparison_2026_05 --model-id anthropic/claude-opus-4.7 --text-id short_instruction --provider-only anthropic --no-provider-fallbacks --force --yes
```

By default, `run-suite` skips a model when saved run-scoped results already
contain every selected `text_id` and language for that `model_id`. Use `--force`
to re-run even when complete saved results exist:

```powershell
uv run lang-token-bench run-suite --suite budget_2026_04 --force --dry-run
```

By default, a failed suite model stops the suite. To record the failure and
continue with later models:

```powershell
uv run lang-token-bench run-suite --suite budget_2026_04 --continue-on-error --yes
```

### Summary And Debug Sources

Summarize saved results for selected models from a benchmark suite:

```powershell
uv run lang-token-bench summarize --suite public_comparison_2026_04
```

To inspect which saved run folders and `results.csv` files were used for each
model in a summary:

```powershell
uv run lang-token-bench summarize --suite all_2026_04 --debug-sources
```

### Plot Generation

Generate figures from a saved suite summary:

```powershell
uv run lang-token-bench plot --suite public_comparison_2026_04
```

### Testing

Run tests:

```powershell
uv run pytest
```

If pytest fails in a Windows/Codex environment while importing `_overlapped`
with `WinError 10106`, run it without pytest's debugging plugin:

```powershell
uv run pytest -p no:debugging
```

## Configuration

Languages are configured in:

```text
configs/languages.yaml
```

Language entries can include `plot_label`; plots use it for axis labels and
fall back to `name` when it is unset.

Models and their intended counters are configured in:

```text
configs/models.yaml
```

Model entries can include `short_name`; plots use `short_name`, then
`display_name`, then `model_id` as the fallback display label.

Benchmark suites are configured in:

```text
configs/benchmark_suites.yaml
```

Suites group model IDs for later validation and summary generation. For
example, `main_2026_04`, `budget_2026_04`, `all_2026_04`, and
`public_comparison_2026_04` can select different model sets without changing
saved run data.

Sample texts are configured in:

```text
datasets/sample_texts.yaml
```

`sample_texts.yaml` supports multiple text records from the beginning:

```yaml
texts:
  - id: short_instruction
    description: A short instruction prompt
    contents:
      en: "Summarize the following text in three bullet points."
      ja: "次の文章を3つの箇条書きで要約してください。"
```

## Environment Variables

The current benchmark workflow uses OpenRouter observed usage as the primary
API-backed measurement path. The only required credential for current API
features is `OPENROUTER_API_KEY`.

The CLI loads a project-root `.env` file at startup when it exists.

The application loads `.env` safely with existing OS environment variables
taking priority. Missing `.env` files are accepted, and key values are never
printed.

`.env.example` documents the expected key names:

```text
OPENROUTER_API_KEY=
```

Anthropic, Gemini, Qwen, Kimi, and other models are compared through
OpenRouter model IDs in the current workflow. Direct Anthropic, Gemini, and
Hugging Face backends are optional future reference backends and do not require
API keys today.

To configure local API credentials, copy `.env.example` to `.env` and fill in
only the keys you need. `.env` is ignored by Git and must never be committed.
Agents and contributors must not inspect, print, or edit `.env` contents.
Do not print API keys in logs, reports, or errors.

## Where To Look After Running

Start with the suite-scoped summary and figures:

- `outputs/summaries/<suite_name>/summary_ratio_by_language_model.md`
- `outputs/summaries/<suite_name>/figures/heatmap_ratio_by_language_model.svg`
- `outputs/summaries/<suite_name>/figures/*.svg`

For run provenance and detailed rows, check:

- `outputs/run_history.csv`
- `outputs/runs/<run_id>/results.csv`
- `outputs/runs/<run_id>/run_summary.json`

## Outputs

The benchmark writes:

```text
outputs/results.csv
outputs/results.md
outputs/run_summary.json
outputs/run_history.csv
outputs/runs/<run_id>/results.csv
outputs/runs/<run_id>/results.md
outputs/runs/<run_id>/run_summary.json
outputs/suite_runs/<suite_run_id>/suite_summary.json
```

The CSV contains one row per model, text, and language. Key fields include:

- `model_id`
- `provider`
- `counter`
- `counting_method`
- `language_code`
- `language_name`
- `text_id`
- `token_count`
- `ratio_to_english`
- `input_price_per_1m_tokens`
- `estimated_input_cost_usd`
- `timestamp_utc`

The Markdown report is a human-readable summary of the same results.

OpenRouter credit balances are displayed by the CLI and written to
`outputs/run_summary.json` plus appended to `outputs/run_history.csv` for
OpenRouter usage runs. Account credit balances are not written to
`outputs/results.csv` or `outputs/results.md`.

For OpenRouter usage runs, the CLI also creates a run directory under
`outputs/runs/<run_id>/`. The `run_id` is based on the UTC timestamp and a
safe filename version of the model ID. The top-level `results.csv` and
`results.md` remain the latest results for convenience.

For `run-suite`, models from the selected suite are executed in order. Each
model gets its own run directory under `outputs/runs/<run_id>/`, and
`outputs/run_history.csv` receives one row per model run. The suite-level
summary is written to `outputs/suite_runs/<suite_run_id>/suite_summary.json`.
It includes requested/completed/skipped/failed models, total rows, run IDs, and
failure reasons, plus credit usage across the suite when OpenRouter credits
were tracked. By default, models with complete saved run-scoped results for the
selected text IDs and languages are skipped to avoid duplicate paid OpenRouter
calls. Use `--force` when a deliberate re-run is needed. Use
`--continue-on-error` only when partial suite completion is acceptable.

Summary commands read saved `results.csv` files from `outputs/runs/*/` and
also include the latest `outputs/results.csv` when it exists.
The top-level summary outputs are kept as latest files:

```text
outputs/summary_ratio_by_language_model.csv
outputs/summary_ratio_by_language_model.md
outputs/heatmap_ratio_language_model.csv
```

Each `summarize --suite <suite_name>` run also writes suite-scoped files under
a safe folder name:

```text
outputs/summaries/<suite_name>/summary_ratio_by_language_model.csv
outputs/summaries/<suite_name>/summary_ratio_by_language_model.md
outputs/summaries/<suite_name>/heatmap_ratio_language_model.csv
```

Summary rows are languages, columns are model IDs, and values are average
`ratio_to_english` across saved text records. English is shown as `1.0` when
results are available. `summary_ratio_by_language_model.csv` and `.md` include
an `Avg` column and final `Avg` row. The `Avg` row excludes English from the
language average, the English row's `Avg` value is `1.0`, and the bottom-right
average excludes English across all models.

`heatmap_ratio_language_model.csv` is written in long format for charting:

```text
language_code,language_name,model_id,ratio_to_english,is_average
```

It includes the same average cells and marks them with `is_average=true`, so a
plotting script can include or filter average annotations explicitly.

The `plot` command reads the suite-scoped
`summary_ratio_by_language_model.csv` and writes figures under:

```text
outputs/summaries/<suite_name>/figures/
```

It always generates `heatmap_ratio_by_language_model.png` and `.svg`. Chart
definitions in `configs/benchmark_suites.yaml` can add extra figures. The
`public_comparison_2026_04` suite includes an OpenAI vs Anthropic two-model bar
chart based on OpenRouter observed usage, written as
`openai_vs_anthropic_bar.png` and `.svg`. After plotting, the
suite-scoped summary Markdown is updated with a `## Figures` section that
embeds SVG files with relative paths. The top-level latest Markdown is updated
with links back to the suite figure directory when it exists.

The heatmap uses a green-to-neutral-to-orange diverging scale centered on
`1.0`, so ratios below English and ratios above English remain visually
distinct. Two-model bar charts sort languages by the lower-average model's
non-English ratios unless the chart configuration specifies another behavior.

## Backends

Implemented in the MVP:

- `simple`: local deterministic baseline counter for smoke tests and workflow checks.
- `openai-tiktoken`: OpenAI tokenizer counting through optional `tiktoken`.
- `openrouter_usage`: observed prompt-token usage through OpenRouter API. This is the primary API-backed workflow. It requires `uv sync --extra openrouter`, `OPENROUTER_API_KEY`, and `--yes` for real API requests.

Stubbed optional future reference backends. These are not used by the current
OpenRouter observed usage workflow and do not require users to configure direct
provider API keys today:

- `anthropic_api`: optional future Anthropic Token Counting API reference backend.
- `gemini_api`: optional future Gemini `countTokens` reference backend.
- `hf_tokenizer`: optional future Hugging Face tokenizer reference backend.
- `playwright_web`: browser-based official tokenizer checks for limited manual validation.

Each backend reports a `counting_method` so observed usage values and official tokenizer-style counts can be compared without mixing their meaning.

## OpenRouter Usage Backend Design

The OpenRouter backend is intentionally separate from official tokenizer counters.

- `counting_method`: `openrouter_usage`
- Credential source: `OPENROUTER_API_KEY` from `os.environ` after CLI startup
  loads `.env` when present.
- Request shape: Chat Completions API with `max_tokens` set from
  `--max-output-tokens`, `temperature: 0`, and the sample text as the user
  message. The default is `16`.
- Optional provider routing: `--provider-only`, `--provider-ignore`,
  `--provider-order`, and `--no-provider-fallbacks` map to OpenRouter's
  request-level `provider` object. Use these options to test whether a
  provider such as `anthropic` succeeds when another route such as
  `amazon-bedrock` is unstable.
- Measured value: `usage.prompt_tokens` from the OpenRouter response.
- Safety rule: OpenRouter models are disabled by default and real API requests require `--yes`.
- Credit tracking: real `openrouter-usage` runs always fetch OpenRouter credits
  before and after the benchmark. If the pre-run credit check fails, the
  benchmark stops before sending usage requests.

OpenRouter usage runs measure prompt tokens, so generated output is kept as
small as practical. The default completion budget is `16` because some
providers routed through OpenRouter reject smaller values even when the prompt
token count is the only benchmark metric.

Use `--dry-run`, `--limit`, `--model-id`, `--text-id`, `--language-code`, and
`--max-output-tokens` before any real usage run. When provider routing is used,
dry-runs print the routing configuration, and OpenRouter run summaries store it
without any API key values.

Credit checks use the OpenRouter Credits API at
`https://openrouter.ai/api/v1/credits`, read `total_credits` and `total_usage`,
and calculate `remaining_credits` as `total_credits - total_usage`.
For benchmark runs, `credits_used` is calculated as
`credits_before_remaining - credits_after_remaining`.

Model validation uses the OpenRouter Models API at
`https://openrouter.ai/api/v1/models` to check whether suite model IDs exist.
It does not run Chat Completions and does not measure token usage.

## Roadmap

- Add optional API-backed integration tests guarded by explicit opt-in flags.
- Keep OpenRouter observed usage as the primary comparison path.
- Consider direct Anthropic Token Counting API support later as a low-priority reference backend.
- Consider direct Gemini `countTokens` API support later as a low-priority reference backend.
- Consider Hugging Face tokenizer support later as a low-priority reference backend.
- Add Playwright-based official tokenizer page checks.
- Expand multilingual sample datasets.
- Add richer report grouping and visualization.

## License

Apache License 2.0. See [LICENSE](LICENSE).
