from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    name: str
    native_name: str
    enabled: bool
    plot_label: Optional[str] = None


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    display_name: str
    counter: str
    tokenizer_name: Optional[str]
    input_price_per_1m_tokens: Optional[float]
    enabled: bool
    short_name: Optional[str] = None


@dataclass(frozen=True)
class SampleText:
    id: str
    description: str
    contents: dict[str, str]


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    name: str
    description: str
    model_ids: list[str]
    charts: list[ChartConfig]


@dataclass(frozen=True)
class ChartConfig:
    id: str
    type: str
    title: str
    model_ids: list[str]
    output_name: str
    sort_languages: str = "by_lower_average_model"
    show_value_labels: bool = True
    legend_position: str = "top"


@dataclass(frozen=True)
class TokenCountResult:
    token_count: int
    counter: str
    counting_method: str
    model_id: Optional[str] = None
    tokenizer_name: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkResult:
    model_id: str
    provider: str
    counter: str
    counting_method: str
    language_code: str
    language_name: str
    text_id: str
    token_count: int
    ratio_to_english: Optional[float]
    input_price_per_1m_tokens: Optional[float]
    estimated_input_cost_usd: Optional[float]
    timestamp_utc: str


@dataclass(frozen=True)
class BenchmarkPlanItem:
    model: ModelConfig
    sample_text: SampleText
    language: LanguageConfig
