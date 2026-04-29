from __future__ import annotations

from typing import Optional


def estimate_input_cost_usd(
    token_count: int,
    input_price_per_1m_tokens: Optional[float],
) -> Optional[float]:
    if input_price_per_1m_tokens is None:
        return None
    return round((token_count / 1_000_000) * input_price_per_1m_tokens, 10)

