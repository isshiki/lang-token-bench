from __future__ import annotations

from lang_token_bench.counters.simple_counter import SimpleCounter


def test_simple_counter_counts_deterministically() -> None:
    counter = SimpleCounter()

    first = counter.count("Hello world!")
    second = counter.count("Hello world!")

    assert first.token_count == second.token_count
    assert first.token_count == 3
    assert first.counter == "simple"
    assert first.counting_method == "simple_regex_baseline"


def test_simple_counter_handles_cjk_text() -> None:
    counter = SimpleCounter()

    result = counter.count("次の文章を要約してください。")

    assert result.token_count > 1

