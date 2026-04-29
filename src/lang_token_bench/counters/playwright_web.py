from __future__ import annotations

from lang_token_bench.counters.base import TokenCounter
from lang_token_bench.schema import ModelConfig, TokenCountResult


class PlaywrightWebCounter(TokenCounter):
    name = "playwright-web"
    counting_method = "browser_check"

    def count(
        self,
        text: str,
        model: ModelConfig | None = None,
    ) -> TokenCountResult:
        raise NotImplementedError(
            "playwright_web counter is not implemented yet. "
            "It is reserved for limited browser checks against official tokenizer pages."
        )

