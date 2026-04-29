from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from lang_token_bench.openrouter_credits import OpenRouterCredits, format_credit_amount


@dataclass(frozen=True)
class OpenRouterCreditRunSummary:
    run_id: str
    started_at_utc: str
    ended_at_utc: str
    counter: str
    model_id: str
    text_id: str
    rows_executed: int
    credits_before: OpenRouterCredits
    credits_after: OpenRouterCredits

    @property
    def credits_used(self) -> Decimal:
        return self.credits_before.remaining_credits - self.credits_after.remaining_credits


@dataclass(frozen=True)
class SuiteRunSummary:
    suite_run_id: str
    suite_name: str
    started_at_utc: str
    ended_at_utc: str
    models_requested: list[str]
    models_completed: list[str]
    models_skipped: list[str]
    models_failed: list[str]
    failure_reasons: dict[str, str]
    total_rows_executed: int
    credits_before_remaining: Decimal | None
    credits_after_remaining: Decimal | None
    run_ids: list[str]

    @property
    def credits_used(self) -> Decimal | None:
        if self.credits_before_remaining is None or self.credits_after_remaining is None:
            return None
        return self.credits_before_remaining - self.credits_after_remaining


RUN_SUMMARY_FIELDS = [
    "run_id",
    "started_at_utc",
    "ended_at_utc",
    "counter",
    "model_id",
    "text_id",
    "rows_executed",
    "credits_before_total",
    "credits_before_usage",
    "credits_before_remaining",
    "credits_after_total",
    "credits_after_usage",
    "credits_after_remaining",
    "credits_used",
]

RUN_HISTORY_FIELDS = [
    "run_id",
    "started_at_utc",
    "ended_at_utc",
    "counter",
    "model_id",
    "text_id",
    "rows_executed",
    "credits_before_remaining",
    "credits_after_remaining",
    "credits_used",
]


def build_run_id(started_at_utc: str, model_ids: list[str]) -> str:
    timestamp = (
        started_at_utc.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace("Z", "")
    )
    model_part = "__".join(_safe_filename_part(model_id) for model_id in model_ids)
    return f"{timestamp}_{model_part}"


def build_suite_run_id(started_at_utc: str, suite_name: str) -> str:
    return build_run_id(started_at_utc, [suite_name])


def write_run_summary(summary: OpenRouterCreditRunSummary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_summary_row(summary), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def write_suite_summary(summary: SuiteRunSummary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_suite_summary_row(summary), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def append_run_history(summary: OpenRouterCreditRunSummary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: _summary_row(summary)[key] for key in RUN_HISTORY_FIELDS})
    return path


def _suite_summary_row(summary: SuiteRunSummary) -> dict[str, object]:
    return {
        "suite_run_id": summary.suite_run_id,
        "suite_name": summary.suite_name,
        "started_at_utc": summary.started_at_utc,
        "ended_at_utc": summary.ended_at_utc,
        "models_requested": summary.models_requested,
        "models_completed": summary.models_completed,
        "models_skipped": summary.models_skipped,
        "models_failed": summary.models_failed,
        "failure_reasons": summary.failure_reasons,
        "total_rows_executed": summary.total_rows_executed,
        "credits_before_remaining": _format_optional_credit(
            summary.credits_before_remaining
        ),
        "credits_after_remaining": _format_optional_credit(
            summary.credits_after_remaining
        ),
        "credits_used": _format_optional_credit(summary.credits_used),
        "run_ids": summary.run_ids,
    }


def _format_optional_credit(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format_credit_amount(value)


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    safe = safe.strip("-._")
    return safe or "unknown"


def _summary_row(summary: OpenRouterCreditRunSummary) -> dict[str, str | int]:
    return {
        "run_id": summary.run_id,
        "started_at_utc": summary.started_at_utc,
        "ended_at_utc": summary.ended_at_utc,
        "counter": summary.counter,
        "model_id": summary.model_id,
        "text_id": summary.text_id,
        "rows_executed": summary.rows_executed,
        "credits_before_total": format_credit_amount(summary.credits_before.total_credits),
        "credits_before_usage": format_credit_amount(summary.credits_before.total_usage),
        "credits_before_remaining": format_credit_amount(summary.credits_before.remaining_credits),
        "credits_after_total": format_credit_amount(summary.credits_after.total_credits),
        "credits_after_usage": format_credit_amount(summary.credits_after.total_usage),
        "credits_after_remaining": format_credit_amount(summary.credits_after.remaining_credits),
        "credits_used": format_credit_amount(summary.credits_used),
    }
