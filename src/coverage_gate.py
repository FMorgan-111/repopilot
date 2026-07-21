"""Dynamic differential regression coverage proof.

Discovery is intentionally advisory.  Only four stable executions performed by
``validate_differential_coverage`` can produce a verified decision.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from .model_provider import redact_secrets
from .repo_paths import canonical_repo_path
from .state import (
    AgentState,
    CoverageStatus,
    TestRunFingerprint,
)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_OPTION_RE = re.compile(
    r"(?:-[qvxs]+|--disable-warnings|--tb=(?:auto|long|short|line|native|no)|"
    r"--maxfail=[0-9]+)"
)
_TEST_NAME_RE = re.compile(r"(?:test_.+|.+_test)\.py\Z", re.IGNORECASE)
_EVALUATOR_ONLY_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gold[_ -]?patch|test[_ -]?patch|"
    r"fail[_ -]?to[_ -]?pass|pass[_ -]?to[_ -]?pass|instance[_ -]?id|"
    r"evaluator|grading|oracle)(?![A-Za-z0-9])"
)
_FAILED_PYTEST_RE = re.compile(
    r"(?m)^(?:FAILED|ERROR)\s+(.+?(?:::[^\s]+)+)(?:\s+-|\s*$)"
)
_FAILED_UNITTEST_RE = re.compile(r"(?m)^(?:FAIL|ERROR):\s+([^\s(]+)(?:\s+\(([^)]+)\))?")
_INFRA_MARKERS = (
    "error collecting",
    "importerror",
    "modulenotfounderror",
    "fixture '",
    "fixture \"",
    "error at setup",
    "internalerror",
    "segmentation fault",
    "core dumped",
    "timed out",
    "timeout",
    "keyboardinterrupt",
)
_ASSERTION_MARKERS = (
    "assertionerror",
    "assert ",
    "expected",
    "actual",
    " got ",
    " != ",
    " == ",
)
_MAX_TEST_BYTES = 512_000


class ChangedTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    symbols: list[str] = Field(default_factory=list)


class CoverageCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_files: list[str] = Field(min_length=1, max_length=16)
    argv: list[str] = Field(min_length=2, max_length=64)
    source: Literal["existing", "generated"]


class CoverageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verified: bool
    status: CoverageStatus
    reason: str
    candidate: CoverageCandidate | None = None
    fixed_runs: list[TestRunFingerprint] = Field(default_factory=list)
    base_runs: list[TestRunFingerprint] = Field(default_factory=list)


def _git(root: Path, argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _exact_repo_root(repo_path: Path) -> Path:
    root = repo_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository path is not a directory")
    result = _git(
        root,
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise ValueError("repository path is not an exact Git root")
    return root


def _exact_ref(root: Path, base_ref: str) -> str:
    if not isinstance(base_ref, str) or not _COMMIT_RE.fullmatch(base_ref):
        raise ValueError("base ref must be an exact lowercase 40-character SHA")
    resolved = _git(
        root,
        ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        check=False,
    )
    if resolved.returncode or resolved.stdout.strip() != base_ref:
        raise ValueError("base ref does not resolve to the exact commit")
    return base_ref


def _symbol_spans(source: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            nested = prefix
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = (*prefix, child.name)
                spans.append(
                    (
                        child.lineno,
                        getattr(child, "end_lineno", child.lineno),
                        ".".join(nested),
                    )
                )
            visit(child, nested)

    visit(tree)
    return spans


def collect_changed_targets(repo_path: Path, base_ref: str) -> list[ChangedTarget]:
    """Collect changed production files and Python symbols from zero-context diff."""
    root = _exact_repo_root(repo_path)
    ref = _exact_ref(root, base_ref)
    result = _git(
        root,
        ["git", "diff", "--unified=0", ref, "--"],
        check=False,
    )
    if result.returncode:
        raise RuntimeError("changed-target diff failed")

    changed_lines: dict[str, set[int]] = {}
    current = ""
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            raw = line[6:]
            try:
                current = canonical_repo_path(raw)
            except ValueError:
                current = ""
            if current:
                changed_lines.setdefault(current, set())
            continue
        if not current or not line.startswith("@@"):
            continue
        match = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        changed_lines[current].update(range(start, start + max(1, count)))

    targets: list[ChangedTarget] = []
    for file_path in sorted(changed_lines):
        if is_allowed_test_path(root, file_path):
            continue
        path = root / file_path
        symbols: list[str] = []
        if path.suffix == ".py" and path.is_file() and not path.is_symlink():
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                source = ""
            lines = changed_lines[file_path]
            symbols = sorted(
                {
                    symbol
                    for start, end, symbol in _symbol_spans(source)
                    if any(start <= line <= end for line in lines)
                }
            )
        targets.append(ChangedTarget(file_path=file_path, symbols=symbols))
    return targets


def _path_has_symlink(root: Path, relative: str) -> bool:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            break
    return False


def is_allowed_test_path(repo_root: Path, file_path: str) -> bool:
    """Return whether a canonical path fits an established Python test layout."""
    try:
        root = repo_root.resolve(strict=True)
        relative = canonical_repo_path(file_path)
    except (OSError, ValueError):
        return False
    if (
        not root.is_dir()
        or _EVALUATOR_ONLY_RE.search(relative)
        or not _TEST_NAME_RE.fullmatch(Path(relative).name)
    ):
        return False
    if _path_has_symlink(root, relative):
        return False
    destination = (root / relative).resolve(strict=False)
    if not destination.is_relative_to(root):
        return False
    if destination.exists() and not destination.is_file():
        return False

    parts = Path(relative).parts
    test_root_index = next(
        (index for index, part in enumerate(parts[:-1]) if part.casefold() in {"test", "tests"}),
        None,
    )
    if test_root_index is not None:
        established = root.joinpath(*parts[: test_root_index + 1])
        return established.is_dir() and not established.is_symlink()

    # Root-level or colocated tests are accepted only when that convention is
    # already present in the same directory.
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        return False
    return any(
        item != destination
        and item.is_file()
        and not item.is_symlink()
        and _TEST_NAME_RE.fullmatch(item.name)
        for item in parent.iterdir()
    ) or destination.is_file()


def _candidate_test_files(argv: list[str]) -> list[str]:
    prefix = 1 if argv[:1] == ["pytest"] else 3 if argv[:3] == ["python", "-m", "pytest"] else 0
    if not prefix or len(argv) <= prefix:
        raise ValueError("candidate command is not an authorized pytest form")
    selectors: list[str] = []
    for token in argv[prefix:]:
        if not isinstance(token, str) or not token or "\0" in token:
            raise ValueError("candidate command contains an invalid token")
        if token.startswith("-"):
            if not _SAFE_OPTION_RE.fullmatch(token):
                raise ValueError("candidate command contains an unsafe option")
            continue
        raw_path = token.split("::", 1)[0]
        selectors.append(canonical_repo_path(raw_path))
    if not selectors:
        raise ValueError("candidate command has no targeted test selector")
    return list(dict.fromkeys(selectors))


def _validate_candidate(root: Path, candidate: CoverageCandidate) -> None:
    paths = [canonical_repo_path(item) for item in candidate.test_files]
    if len(paths) != len(set(paths)):
        raise ValueError("candidate test files must be unique")
    if not all(is_allowed_test_path(root, path) for path in paths):
        raise ValueError("candidate contains a non-test or unsafe path")
    if _candidate_test_files(candidate.argv) != paths:
        raise ValueError("candidate argv does not exactly target its test files")


def _authorize_candidate_files(
    state: AgentState,
    root: Path,
    ref: str,
    candidate: CoverageCandidate,
) -> None:
    if candidate.source == "generated":
        approval = state.tool_patch_approval
        approved = {item.path: item for item in state.generated_test_approvals}
        if approval is None:
            raise ValueError("generated candidate lacks PatchGate approval")
        for file_path in candidate.test_files:
            item = approved.get(file_path)
            path = root / file_path
            if (
                file_path not in state.coverage_test_files
                or item is None
                or item.patch_gate_fingerprint != approval.patch_gate_fingerprint
                or not path.is_file()
                or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != item.content_sha256
            ):
                raise ValueError("generated candidate approval does not match its file")
        return

    for file_path in candidate.test_files:
        source = root / file_path
        base = subprocess.run(
            ["git", "show", f"{ref}:{file_path}"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if (
            base.returncode
            or not source.is_file()
            or source.is_symlink()
            or source.read_bytes() != base.stdout
        ):
            raise ValueError("existing candidate differs from its exact base file")


def _candidate_from_command(
    root: Path,
    command: str,
    *,
    generated: set[str],
) -> CoverageCandidate | None:
    try:
        argv = shlex.split(command, posix=True)
        files = _candidate_test_files(argv)
        source: Literal["existing", "generated"] = (
            "generated" if any(path in generated for path in files) else "existing"
        )
        candidate = CoverageCandidate(test_files=files, argv=argv, source=source)
        _validate_candidate(root, candidate)
        return candidate
    except (ValueError, TypeError):
        return None


def discover_coverage_candidates(
    state: AgentState,
    targets: list[ChangedTarget],
) -> list[CoverageCandidate]:
    """Find plausible targeted tests without treating references as proof."""
    root = _exact_repo_root(Path(state.repo_path))
    generated = set(state.coverage_test_files)
    candidates: list[CoverageCandidate] = []
    if state.test_command:
        candidate = _candidate_from_command(root, state.test_command, generated=generated)
        if candidate is not None:
            candidates.append(candidate)

    symbols = {symbol.rsplit(".", 1)[-1] for target in targets for symbol in target.symbols}
    if symbols:
        tracked = _git(
            root,
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            check=False,
        )
        if not tracked.returncode:
            for raw in tracked.stdout.split("\0"):
                if not raw or not is_allowed_test_path(root, raw):
                    continue
                path = root / raw
                try:
                    if path.stat().st_size > _MAX_TEST_BYTES:
                        continue
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if not any(re.search(rf"\b{re.escape(symbol)}\b", text) for symbol in symbols):
                    continue
                source: Literal["existing", "generated"] = (
                    "generated" if raw in generated else "existing"
                )
                candidate = CoverageCandidate(
                    test_files=[raw],
                    argv=["python", "-m", "pytest", raw, "-q"],
                    source=source,
                )
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates


def _normalize_runtime_text(output: str, repo_path: Path) -> str:
    text = output.replace("\\", "/")
    root = str(repo_path.resolve()).replace("\\", "/")
    text = text.replace(root, "<repo>")
    text = re.sub(r"(?:/private)?/tmp/[^\s:]+", "<tmp>", text)
    text = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b", "<runtime-id>", text)
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", "<address>", text)
    text = re.sub(r":\d+(?=[:\s])", ":<line>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", text)
    text = re.sub(r"\bin \d+(?:\.\d+)? seconds?\b", "in <duration>", text)
    return text


def _normalize_test_id(test_id: str, repo_path: Path) -> str:
    normalized = redact_secrets(test_id.replace("\\", "/").strip())
    root = str(repo_path.resolve()).replace("\\", "/").rstrip("/") + "/"
    if normalized.startswith(root):
        normalized = normalized[len(root) :]
    normalized = re.sub(r"^(?:/private)?/tmp/[^\s:]+/", "", normalized)
    return _EVALUATOR_ONLY_RE.sub("[REDACTED_EVALUATOR_FIELD]", normalized)


def normalize_test_result(
    exit_code: int,
    output: str,
    repo_path: Path,
) -> TestRunFingerprint:
    """Reduce raw test output to stable, safe assertion evidence."""
    if exit_code == 0:
        return TestRunFingerprint(exit_code=0, outcome="pass", summary="pass")

    lowered = output.casefold()
    pytest_ids = [_normalize_test_id(item, repo_path) for item in _FAILED_PYTEST_RE.findall(output)]
    unittest_ids = []
    for name, owner in _FAILED_UNITTEST_RE.findall(output):
        unittest_ids.append(f"{owner}.{name}" if owner else name)
    failing_ids = sorted(set([*pytest_ids, *unittest_ids]))
    has_assertion = any(marker in lowered for marker in _ASSERTION_MARKERS)
    has_infra = any(marker in lowered for marker in _INFRA_MARKERS)
    if not failing_ids or has_infra or not has_assertion:
        return TestRunFingerprint(
            exit_code=exit_code,
            outcome="infra",
            failing_test_ids=failing_ids,
            summary=f"infra; failing_test_count={len(failing_ids)}",
        )

    normalized = _normalize_runtime_text(output, repo_path)
    assertion_lines: list[str] = []
    for line in normalized.splitlines():
        if not any(marker in line.casefold() for marker in _ASSERTION_MARKERS):
            continue
        compact = " ".join(line.split())
        if compact and compact not in assertion_lines:
            assertion_lines.append(compact)
    payload = json.dumps(
        {"failing_test_ids": failing_ids, "assertions": assertion_lines[:16]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return TestRunFingerprint(
        exit_code=exit_code,
        outcome="assertion_failure",
        failing_test_ids=failing_ids,
        assertion_fingerprint=fingerprint,
        summary=(
            f"assertion_failure; failing_test_ids={','.join(failing_ids)}; "
            f"fingerprint={fingerprint[:16]}"
        )[:500],
    )


@contextmanager
def temporary_base_checkout(repo_path: Path, base_ref: str) -> Iterator[Path]:
    """Create and always remove an exact detached temporary Git worktree."""
    root = _exact_repo_root(repo_path)
    ref = _exact_ref(root, base_ref)
    parent = root.parent.resolve()
    checkout = Path(tempfile.mkdtemp(prefix=".repopilot-coverage-", dir=parent))
    checkout.rmdir()  # git worktree requires the target not to exist
    added = False

    def validated_path() -> Path:
        if (
            checkout.parent.resolve() != parent
            or not checkout.name.startswith(".repopilot-coverage-")
            or checkout == root
            or checkout.is_symlink()
        ):
            raise RuntimeError("temporary checkout path validation failed")
        return checkout

    try:
        _git(
            root,
            ["git", "worktree", "add", "--detach", str(validated_path()), ref],
        )
        added = True
        yield checkout
    finally:
        validated = validated_path()
        if added:
            _git(
                root,
                ["git", "worktree", "remove", "--force", str(validated)],
                check=False,
            )
        if validated.exists():
            shutil.rmtree(validated)
        _git(root, ["git", "worktree", "prune"], check=False)


async def _run_candidate(root: Path, argv: list[str]) -> TestRunFingerprint:
    def run() -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            argv,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )

    try:
        result = await asyncio.to_thread(run)
    except (OSError, subprocess.SubprocessError):
        return TestRunFingerprint(
            exit_code=-1,
            outcome="infra",
            summary="infra; execution_failed",
        )
    return normalize_test_result(
        result.returncode,
        f"{result.stdout}\n{result.stderr}",
        root,
    )


def _decision(
    candidate: CoverageCandidate,
    reason: str,
    *,
    fixed_runs: list[TestRunFingerprint] | None = None,
    base_runs: list[TestRunFingerprint] | None = None,
) -> CoverageDecision:
    return CoverageDecision(
        verified=False,
        status="failed",
        reason=reason,
        candidate=candidate,
        fixed_runs=fixed_runs or [],
        base_runs=base_runs or [],
    )


def _copy_test_to_base(fixed_root: Path, base_root: Path, file_path: str) -> None:
    relative = canonical_repo_path(file_path)
    source = fixed_root / relative
    destination = base_root / relative
    if (
        _path_has_symlink(fixed_root, relative)
        or not source.is_file()
        or source.is_symlink()
        or source.stat().st_size > _MAX_TEST_BYTES
    ):
        raise ValueError("candidate test source is not an authorized regular file")
    if _path_has_symlink(base_root, relative):
        raise ValueError("candidate test destination crosses a symlink")
    resolved = destination.resolve(strict=False)
    if not resolved.is_relative_to(base_root.resolve()):
        raise ValueError("candidate test destination escaped base checkout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


async def validate_differential_coverage(
    state: AgentState,
    candidate: CoverageCandidate,
) -> CoverageDecision:
    """Require fixed pass twice and the exact same base assertion twice."""
    try:
        root = _exact_repo_root(Path(state.repo_path))
        ref = _exact_ref(root, state.repo_ref)
        head = _git(root, ["git", "rev-parse", "HEAD"]).stdout.strip()
        if head != ref:
            raise ValueError("fixed checkout HEAD does not match state.repo_ref")
        _validate_candidate(root, candidate)
        _authorize_candidate_files(state, root, ref, candidate)
    except (OSError, RuntimeError, ValueError):
        return _decision(candidate, "coverage_infra")

    fixed_runs = [await _run_candidate(root, list(candidate.argv)) for _ in range(2)]
    if any(run.outcome != "pass" for run in fixed_runs):
        return _decision(candidate, "fixed_does_not_pass", fixed_runs=fixed_runs)

    base_runs: list[TestRunFingerprint] = []
    try:
        with temporary_base_checkout(root, ref) as base_root:
            for file_path in candidate.test_files:
                _copy_test_to_base(root, base_root, file_path)
            base_runs = [
                await _run_candidate(base_root, list(candidate.argv)) for _ in range(2)
            ]
    except (OSError, RuntimeError, ValueError):
        return _decision(candidate, "coverage_infra", fixed_runs=fixed_runs)

    if any(run.outcome == "infra" for run in base_runs):
        return _decision(
            candidate,
            "coverage_infra",
            fixed_runs=fixed_runs,
            base_runs=base_runs,
        )
    if all(run.outcome == "pass" for run in base_runs):
        return _decision(
            candidate,
            "base_also_passes",
            fixed_runs=fixed_runs,
            base_runs=base_runs,
        )
    if any(run.outcome != "assertion_failure" for run in base_runs):
        return _decision(
            candidate,
            "coverage_infra",
            fixed_runs=fixed_runs,
            base_runs=base_runs,
        )
    stable = (
        base_runs[0].failing_test_ids == base_runs[1].failing_test_ids
        and bool(base_runs[0].failing_test_ids)
        and base_runs[0].assertion_fingerprint
        == base_runs[1].assertion_fingerprint
        and bool(base_runs[0].assertion_fingerprint)
    )
    if not stable:
        return _decision(
            candidate,
            "unstable_base_failure",
            fixed_runs=fixed_runs,
            base_runs=base_runs,
        )
    return CoverageDecision(
        verified=True,
        status=(
            "generated_verified" if candidate.source == "generated" else "existing_verified"
        ),
        reason="verified",
        candidate=candidate,
        fixed_runs=fixed_runs,
        base_runs=base_runs,
    )


__all__ = [
    "ChangedTarget",
    "CoverageCandidate",
    "CoverageDecision",
    "collect_changed_targets",
    "discover_coverage_candidates",
    "is_allowed_test_path",
    "normalize_test_result",
    "temporary_base_checkout",
    "validate_differential_coverage",
]
