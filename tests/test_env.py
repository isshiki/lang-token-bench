from __future__ import annotations

import os
from pathlib import Path

import pytest

from lang_token_bench.env import load_project_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_project_env_loads_temp_dotenv_without_output(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=dummy-test-key\n", encoding="utf-8")

    loaded = load_project_env(tmp_path)

    captured = capsys.readouterr()
    assert loaded is True
    assert os.environ["OPENROUTER_API_KEY"] == "dummy-test-key"
    assert captured.out == ""
    assert captured.err == ""
    assert "dummy-test-key" not in captured.out
    assert "dummy-test-key" not in captured.err


def test_load_project_env_does_not_override_existing_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing-env-value")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=dotenv-value\n", encoding="utf-8")

    loaded = load_project_env(tmp_path)

    assert loaded is True
    assert os.environ["OPENROUTER_API_KEY"] == "existing-env-value"


def test_load_project_env_missing_file_is_ok(tmp_path) -> None:
    assert load_project_env(tmp_path) is False


def test_env_example_only_lists_current_openrouter_key() -> None:
    lines = [
        line.strip()
        for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert lines == ["OPENROUTER_API_KEY="]
