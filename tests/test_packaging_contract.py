from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
_FIXED_EVAL_ID_FILES = (
    "eval/baseline_10_ids.txt",
    "eval/baseline_50_ids.txt",
    "eval/checkpoint_5_ids.txt",
)


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


def test_fixed_eval_id_contract_files_are_tracked():
    for relative_path in _FIXED_EVAL_ID_FILES:
        assert (ROOT / relative_path).is_file()

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *_FIXED_EVAL_ID_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert tracked.returncode == 0, tracked.stderr


def test_fixed_eval_id_contract_files_are_not_ignored():
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *_FIXED_EVAL_ID_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert ignored.returncode == 1, ignored.stdout


def test_base_install_can_use_disabled_episode_memory_without_numpy():
    code = """
import builtins
import sys

real_import = builtins.__import__

def reject_optional_memory_dependencies(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'numpy', 'sqlite_vec'}:
        raise ImportError(f'blocked optional dependency: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_optional_memory_dependencies
from src.memory.error_episode_store import get_episode_store
assert get_episode_store() is None
assert 'numpy' not in sys.modules
"""
    env = os.environ.copy()
    env.pop("REPOPILOT_ENABLE_EPISODES", None)

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
