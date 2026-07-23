from __future__ import annotations

import os
import re
import shutil
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
_EVAL_REQUIREMENT_FILES = (
    "requirements-eval.in",
    "requirements-eval.lock",
)


def _dependency_names(dependencies: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower()
        for dependency in dependencies
    }


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _locked_requirement_records(path: Path) -> list[list[str]]:
    records: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line == raw_line.lstrip():
            records.append([raw_line])
        else:
            assert records, f"orphaned lock continuation: {raw_line}"
            records[-1].append(raw_line)
    return records


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


def test_eval_requirement_inputs_are_tracked_release_files() -> None:
    for relative_path in _EVAL_REQUIREMENT_FILES:
        assert (ROOT / relative_path).is_file()

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *_EVAL_REQUIREMENT_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, tracked.stderr

    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *_EVAL_REQUIREMENT_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 1, ignored.stdout


def test_eval_requirement_input_explicitly_covers_all_workflow_dependencies() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    declared = _dependency_names(
        _requirement_lines(ROOT / "requirements-eval.in")
    )
    required = _dependency_names(
        project["dependencies"]
        + project["optional-dependencies"]["memory"]
        + project["optional-dependencies"]["eval"]
        + metadata["build-system"]["requires"]
    )

    assert required | {"wheel"} <= declared
    assert not any(
        line.startswith(("-e ", ".", "-r ", "--requirement "))
        for line in _requirement_lines(ROOT / "requirements-eval.in")
    )


def test_eval_lock_is_fully_pinned_and_hash_covered() -> None:
    records = _locked_requirement_records(ROOT / "requirements-eval.lock")
    assert records

    locked_names: set[str] = set()
    for record in records:
        requirement = record[0].removesuffix("\\").strip()
        match = re.fullmatch(
            r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==[^\s;\\]+"
            r"(?:\s*;\s*[^\\]+)?",
            requirement,
        )
        assert match is not None, f"lock entry is not exactly pinned: {record[0]}"
        locked_names.add(match.group("name").lower().replace("_", "-"))

        rendered = " ".join(line.strip() for line in record)
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})(?:\s|\\|$)", rendered)
        assert hashes, f"lock entry has no SHA-256 hash: {record[0]}"
        without_hashes = re.sub(
            r"\s*\\?\s*--hash=sha256:[0-9a-f]{64}", "", rendered
        ).replace("\\", "").strip()
        assert without_hashes == requirement

    input_names = {
        name.replace("_", "-")
        for name in _dependency_names(
            _requirement_lines(ROOT / "requirements-eval.in")
        )
    }
    assert input_names <= locked_names


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


def test_shipped_src_imports_state_and_cli_without_eval_package(
    tmp_path: Path,
) -> None:
    shutil.copytree(
        ROOT / "src",
        tmp_path / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    code = """
from pathlib import Path
import os
import sys

sys.path.insert(0, os.environ["REPOPILOT_TEST_SHIPPED_ROOT"])
import src
import src.state
import src.cli

source_root = Path(os.environ["REPOPILOT_TEST_SOURCE_ROOT"]).resolve()
assert source_root not in {
    Path(entry).resolve() for entry in __import__("sys").path if entry
}
assert Path(src.__file__).resolve().is_relative_to(
    Path(os.environ["REPOPILOT_TEST_SHIPPED_ROOT"]).resolve()
)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["REPOPILOT_TEST_SOURCE_ROOT"] = str(ROOT)
    env["REPOPILOT_TEST_SHIPPED_ROOT"] = str(tmp_path)

    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_documented_eval_report_script_runs_without_installed_project(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, "-S", str(ROOT / "eval" / "report.py"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_importing_eval_harness_does_not_load_dotenv():
    code = """
import dotenv

def reject_import_time_dotenv(*args, **kwargs):
    raise AssertionError('eval harness loaded dotenv during import')

dotenv.load_dotenv = reject_import_time_dotenv
import eval.harness
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
