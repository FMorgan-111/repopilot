from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _dependency_names(dependencies: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower()
        for dependency in dependencies
    }


def test_memory_and_dev_extras_cover_optional_runtime_and_test_dependencies():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    assert {"fastembed", "numpy", "sqlite-vec"} <= _dependency_names(
        extras["memory"]
    )
    assert {"pytest", "pytest-asyncio", "ruff", "tomli"} <= _dependency_names(
        extras["dev"]
    )


def test_ci_installs_declared_extras_and_runs_tests_and_lint():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert 'pip install -e ".[memory,dev]"' in workflow
    assert "python -m pytest tests/ -q" in workflow
    assert "python -m ruff check src/ tests/ eval/" in workflow
    assert "macos-latest" in workflow


def test_generated_editable_install_metadata_is_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "*.egg-info/" in gitignore
    assert "build/" in gitignore
    assert "dist/" in gitignore
