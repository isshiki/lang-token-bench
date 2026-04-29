from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from lang_token_bench.schema import BenchmarkResult


CSV_FIELDS = [
    "model_id",
    "provider",
    "counter",
    "counting_method",
    "language_code",
    "language_name",
    "text_id",
    "token_count",
    "ratio_to_english",
    "input_price_per_1m_tokens",
    "estimated_input_cost_usd",
    "timestamp_utc",
]


def write_csv_report(results: list[BenchmarkResult], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            row = {
                key: "" if value is None else value
                for key, value in asdict(result).items()
            }
            writer.writerow(row)
    return output_path

