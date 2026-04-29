from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from lang_token_bench.config import PROJECT_ROOT


def load_project_env(project_root: Path = PROJECT_ROOT) -> bool:
    """Load a project-local .env file without overriding existing variables."""
    env_path = project_root / ".env"
    if not env_path.is_file():
        return False
    return bool(load_dotenv(dotenv_path=env_path, override=False))
