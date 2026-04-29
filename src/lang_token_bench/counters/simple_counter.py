from __future__ import annotations

import re

from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


class SimpleCounter(TokenCounter):
    name = "simple"
    counting_method = "simple_regex_baseline"

    _token_pattern = re.compile(
        r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
        r"|\w+|[^\w\s]",
        re.UNICODE,
    )

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        tokens = self._token_pattern.findall(text)
        return TokenCountResult(
            token_count=len(tokens),
            counter=self.name,
            counting_method=self.counting_method,
            model_id=None if model is None else model.id,
            tokenizer_name=None if model is None else model.tokenizer_name,
        )

