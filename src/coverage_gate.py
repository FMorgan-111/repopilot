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
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from .evaluator_safety import sanitize_evaluator_text
from .model_provider import redact_secrets
from .repo_paths import canonical_repo_path
from .safe_subprocess import run_oci_process
from .state import (
    AgentState,
    CoverageProof,
    CoverageStatus,
    SnapshotManifestEntry,
    TestRunFingerprint,
    tool_manifest_fingerprint,
)
from .test_path_policy import is_allowed_test_path, path_has_symlink
from .tool_policy import fixed_test_argv
from .tool_router import disposable_test_snapshot

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_OPTION_RE = re.compile(
    r"(?:-[qvxs]+|--disable-warnings|--tb=(?:auto|long|short|line|native|no)|"
    r"--maxfail=[0-9]+)"
)
_FAILED_PYTEST_RE = re.compile(
    r"(?m)^FAILED\s+(.+?(?:::[^\s]+)+)(?:\s+-|\s*$)"
)
_FAILED_UNITTEST_RE = re.compile(r"(?m)^FAIL:\s+([^\s(]+)(?:\s+\(([^)]+)\))?")
_ERROR_REPORT_RE = re.compile(
    r"(?im)^(?:ERROR(?:\s|:)|.*\berror at (?:setup|teardown)\b)"
)
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


@dataclass(frozen=True)
class ApprovedLiveTarget:
    """One exact UTF-8 file authorized by the current live PatchGate output."""

    path: str
    change: Literal["added", "modified", "deleted"]
    mode: Literal["100644", "100755"]
    content: str | None
    content_sha256: str | None
    size: int
    base_blob_sha: str | None


@dataclass(frozen=True)
class LiveCoverageBinding:
    """Immutable authorization shared by coverage, prediction, and commit."""

    patch_content: str
    patch_sha256: str
    patch_gate_fingerprint: str
    manifest_fingerprint: str
    approved_targets: tuple[ApprovedLiveTarget, ...]


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


def _clear_live_proof(state: AgentState) -> None:
    state.coverage_proof = None
    if state.coverage_status in {"existing_verified", "generated_verified"}:
        state.coverage_status = "pending"
    state.pr_url = None


def _git_bytes(
    root: Path,
    argv: list[str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=root,
        check=False,
        capture_output=True,
        timeout=60,
    )


def _base_blob(
    root: Path,
    ref: str,
    path: str,
) -> tuple[str, str, bytes] | None:
    listed = _git_bytes(
        root,
        ["git", "ls-tree", "-z", ref, "--", f":(literal){path}"],
    )
    if listed.returncode or not listed.stdout:
        return None
    records = [record for record in listed.stdout.split(b"\0") if record]
    if len(records) != 1:
        raise ValueError("base manifest path is ambiguous")
    metadata, separator, raw_path = records[0].partition(b"\t")
    fields = metadata.split()
    if (
        not separator
        or raw_path.decode("utf-8", "strict") != path
        or len(fields) != 3
        or fields[0] not in {b"100644", b"100755"}
        or fields[1] != b"blob"
    ):
        raise ValueError("base manifest entry is not a regular file")
    blob = _git_bytes(root, ["git", "show", f"{ref}:{path}"])
    if blob.returncode:
        raise ValueError("base manifest blob is unavailable")
    return fields[0].decode("ascii"), fields[2].decode("ascii"), blob.stdout


def _changed_live_paths(root: Path, ref: str) -> list[str]:
    tracked = _git_bytes(
        root,
        [
            "git",
            "diff",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "--no-textconv",
            ref,
            "--",
        ],
    )
    untracked = _git_bytes(
        root,
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    if tracked.returncode or untracked.returncode:
        raise RuntimeError("live changed-path enumeration failed")
    paths: set[str] = set()
    folded: set[str] = set()
    for raw in [*tracked.stdout.split(b"\0"), *untracked.stdout.split(b"\0")]:
        if not raw:
            continue
        path = canonical_repo_path(raw.decode("utf-8", "strict"))
        key = path.casefold()
        if key in folded and path not in paths:
            raise ValueError("live changed paths contain a canonical collision")
        folded.add(key)
        paths.add(path)
    return sorted(paths)


def _read_live_regular(root: Path, path: str) -> tuple[str, bytes]:
    current = root
    for component in Path(path).parts:
        current = current / component
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("live changed path contains a symlink")
    info = current.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("live changed path is not a regular file")
    if info.st_size > 512_000_000:
        raise ValueError("live changed file exceeds size limit")
    descriptor = os.open(
        current,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (info.st_dev, info.st_ino):
            raise ValueError("live changed file identity drifted")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 65_536):
            total += len(chunk)
            if total > 512_000_000:
                raise ValueError("live changed file exceeds size limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    text = content.decode("utf-8")
    if "\0" in text:
        raise ValueError("live changed file contains NUL")
    mode: Literal["100644", "100755"] = (
        "100755" if opened.st_mode & 0o111 else "100644"
    )
    return mode, content


def validate_live_coverage_binding(state: AgentState) -> LiveCoverageBinding:
    """Bind the exact live diff and manifest to the persisted PatchGate approval."""
    try:
        root = _exact_repo_root(Path(state.repo_path))
        ref = _exact_ref(root, state.repo_ref)
        head = _git(root, ["git", "rev-parse", "HEAD"]).stdout.strip()
        approval = state.tool_patch_approval
        if head != ref or approval is None or approval.base_ref != ref:
            raise ValueError("live approval base mismatch")
        patch_sha256 = hashlib.sha256(
            state.patch_content.encode("utf-8")
        ).hexdigest()
        if patch_sha256 != approval.patch_sha256:
            raise ValueError("live approval patch fingerprint mismatch")
        manifest_fingerprint = tool_manifest_fingerprint(approval.changed_manifest)
        if manifest_fingerprint != approval.manifest_fingerprint:
            raise ValueError("live approval manifest fingerprint mismatch")
        proof = state.coverage_proof
        if proof is not None and (
            proof.base_ref != ref
            or proof.patch_sha256 != approval.patch_sha256
            or proof.patch_gate_fingerprint != approval.patch_gate_fingerprint
            or proof.manifest_fingerprint != approval.manifest_fingerprint
        ):
            raise ValueError("persisted coverage proof binding mismatch")

        # Import lazily to avoid importing the nodes package while it imports
        # this coverage module during graph construction.
        from .nodes.execute import git_diff

        live_patch = git_diff(str(root))
        if live_patch != state.patch_content:
            raise ValueError("live worktree diff does not match approved patch")

        manifest: list[SnapshotManifestEntry] = []
        targets: list[ApprovedLiveTarget] = []
        for path in _changed_live_paths(root, ref):
            base = _base_blob(root, ref, path)
            live = root / path
            try:
                live.lstat()
            except FileNotFoundError:
                if base is None:
                    raise ValueError("live changed path is missing from base and worktree")
                base_mode, base_sha, base_content = base
                entry = SnapshotManifestEntry(
                    path=path,
                    change="deleted",
                    mode=base_mode,
                    size=len(base_content),
                )
                target = ApprovedLiveTarget(
                    path=path,
                    change="deleted",
                    mode=base_mode,  # type: ignore[arg-type]
                    content=None,
                    content_sha256=None,
                    size=len(base_content),
                    base_blob_sha=base_sha,
                )
            else:
                mode, content = _read_live_regular(root, path)
                digest = hashlib.sha256(content).hexdigest()
                change: Literal["added", "modified"] = (
                    "added" if base is None else "modified"
                )
                entry = SnapshotManifestEntry(
                    path=path,
                    change=change,
                    mode=mode,
                    content_sha256=digest,
                    size=len(content),
                )
                target = ApprovedLiveTarget(
                    path=path,
                    change=change,
                    mode=mode,
                    content=content.decode("utf-8"),
                    content_sha256=digest,
                    size=len(content),
                    base_blob_sha=None if base is None else base[1],
                )
            manifest.append(entry)
            targets.append(target)
        if tuple(manifest) != tuple(approval.changed_manifest):
            raise ValueError("live changed manifest does not match approval")
        return LiveCoverageBinding(
            patch_content=state.patch_content,
            patch_sha256=patch_sha256,
            patch_gate_fingerprint=approval.patch_gate_fingerprint,
            manifest_fingerprint=manifest_fingerprint,
            approved_targets=tuple(targets),
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, UnicodeDecodeError, ValueError):
        _clear_live_proof(state)
        raise ValueError("coverage live binding mismatch") from None


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
            manifest_entry = next(
                (
                    entry
                    for entry in approval.changed_manifest
                    if entry.path == file_path
                ),
                None,
            )
            base = subprocess.run(
                ["git", "show", f"{ref}:{file_path}"],
                cwd=root,
                check=False,
                capture_output=True,
                timeout=30,
            )
            if (
                file_path not in state.coverage_test_files
                or item is None
                or item.patch_gate_fingerprint != approval.patch_gate_fingerprint
                or manifest_entry is None
                or manifest_entry.change != item.change
                or manifest_entry.content_sha256 != item.content_sha256
                or not path.is_file()
                or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != item.content_sha256
            ):
                raise ValueError("generated candidate approval does not match its file")
            if item.change == "added" and base.returncode == 0:
                raise ValueError("added generated candidate exists in exact base")
            if item.change == "modified" and (
                base.returncode != 0
                or hashlib.sha256(base.stdout).hexdigest()
                != item.base_content_sha256
            ):
                raise ValueError("modified generated candidate base preimage mismatch")
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


def coverage_proof_matches_state(state: AgentState, proof: CoverageProof) -> bool:
    """Validate every persisted proof binding against the current approved diff."""
    try:
        root = _exact_repo_root(Path(state.repo_path))
        approval = state.tool_patch_approval
        if approval is None or state.coverage_status != proof.status:
            return False
        if (
            proof.base_ref != state.repo_ref
            or approval.base_ref != state.repo_ref
            or hashlib.sha256(state.patch_content.encode("utf-8")).hexdigest()
            != approval.patch_sha256
            or proof.patch_sha256 != approval.patch_sha256
            or proof.patch_gate_fingerprint != approval.patch_gate_fingerprint
            or proof.manifest_fingerprint != approval.manifest_fingerprint
            or tool_manifest_fingerprint(approval.changed_manifest)
            != approval.manifest_fingerprint
        ):
            return False
        for file_path, expected in proof.test_content_digests.items():
            if not is_allowed_test_path(root, file_path):
                return False
            path = root / file_path
            if (
                path.is_symlink()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                return False
        candidate = CoverageCandidate(
            test_files=proof.test_files,
            argv=proof.argv,
            source=proof.source,
        )
        _validate_candidate(root, candidate)
        _authorize_candidate_files(state, root, state.repo_ref, candidate)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def validate_terminal_coverage_binding(state: AgentState) -> LiveCoverageBinding:
    """Require a semantically valid proof bound to the exact live approval."""
    proof = state.coverage_proof
    if (
        proof is None
        or state.coverage_status
        not in {"existing_verified", "generated_verified"}
        or state.coverage_status != proof.status
    ):
        _clear_live_proof(state)
        raise ValueError("terminal coverage proof is missing or mismatched")
    try:
        semantic_proof = CoverageProof.model_validate(
            proof.model_dump(mode="python")
        )
    except (TypeError, ValueError):
        _clear_live_proof(state)
        raise ValueError("terminal coverage proof semantics are invalid") from None
    binding = validate_live_coverage_binding(state)
    if not coverage_proof_matches_state(state, semantic_proof):
        _clear_live_proof(state)
        raise ValueError("terminal coverage proof binding is invalid")
    return binding


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
    return sanitize_evaluator_text(normalized)


def normalize_test_result(
    exit_code: int,
    output: str,
    repo_path: Path,
) -> TestRunFingerprint:
    """Reduce raw test output to stable, safe assertion evidence."""
    if exit_code == 0:
        return TestRunFingerprint(exit_code=0, outcome="pass", summary="pass")

    lowered = output.casefold()
    pytest_ids = [
        normalized
        for item in _FAILED_PYTEST_RE.findall(output)
        if (normalized := _normalize_test_id(item, repo_path))
    ]
    unittest_ids = []
    for name, owner in _FAILED_UNITTEST_RE.findall(output):
        normalized = _normalize_test_id(f"{owner}.{name}" if owner else name, repo_path)
        if normalized:
            unittest_ids.append(normalized)
    failing_ids = sorted(set([*pytest_ids, *unittest_ids]))
    has_assertion = any(marker in lowered for marker in _ASSERTION_MARKERS)
    has_infra = bool(_ERROR_REPORT_RE.search(output)) or any(
        marker in lowered for marker in _INFRA_MARKERS
    )
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
    raise RuntimeError("repository tests may not execute in a host worktree")


async def _run_isolated_candidate(
    state: AgentState,
    candidate: CoverageCandidate,
    *,
    apply_approved_changes: bool,
) -> TestRunFingerprint:
    validate_live_coverage_binding(state)
    root = Path(state.repo_path).resolve(strict=True)
    config = state.tool_sandbox_config
    if config is None:
        raise ValueError("coverage requires a configured OCI sandbox")
    argv, _normalized = fixed_test_argv(root, state, candidate.argv)
    with disposable_test_snapshot(
        state,
        apply_approved_changes=apply_approved_changes,
        test_overlay_paths=(
            candidate.test_files
            if candidate.source == "generated" and not apply_approved_changes
            else None
        ),
    ) as sandbox:
        result = await asyncio.to_thread(
            run_oci_process,
            argv,
            sandbox=sandbox,
            config=config,
            timeout=300,
            max_output_bytes=8_000,
        )
        fingerprint = normalize_test_result(
            result.returncode,
            f"{result.stdout}\n{result.stderr}",
            sandbox.workspace,
        )
    validate_live_coverage_binding(state)
    return fingerprint


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
        path_has_symlink(fixed_root, relative)
        or not source.is_file()
        or source.is_symlink()
        or source.stat().st_size > _MAX_TEST_BYTES
    ):
        raise ValueError("candidate test source is not an authorized regular file")
    if path_has_symlink(base_root, relative):
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
        validate_live_coverage_binding(state)
        root = _exact_repo_root(Path(state.repo_path))
        ref = _exact_ref(root, state.repo_ref)
        head = _git(root, ["git", "rev-parse", "HEAD"]).stdout.strip()
        if head != ref:
            raise ValueError("fixed checkout HEAD does not match state.repo_ref")
        _validate_candidate(root, candidate)
        _authorize_candidate_files(state, root, ref, candidate)
    except (OSError, RuntimeError, ValueError):
        return _decision(candidate, "coverage_infra")

    try:
        fixed_runs = [
            await _run_isolated_candidate(
                state,
                candidate,
                apply_approved_changes=True,
            )
            for _ in range(2)
        ]
    except (OSError, RuntimeError, ValueError):
        return _decision(candidate, "coverage_infra")
    if any(run.outcome != "pass" for run in fixed_runs):
        return _decision(candidate, "fixed_does_not_pass", fixed_runs=fixed_runs)

    base_runs: list[TestRunFingerprint] = []
    try:
        base_runs = [
            await _run_isolated_candidate(
                state,
                candidate,
                apply_approved_changes=False,
            )
            for _ in range(2)
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
    try:
        validate_live_coverage_binding(state)
    except ValueError:
        return _decision(
            candidate,
            "coverage_infra",
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
    "ApprovedLiveTarget",
    "LiveCoverageBinding",
    "collect_changed_targets",
    "coverage_proof_matches_state",
    "discover_coverage_candidates",
    "is_allowed_test_path",
    "normalize_test_result",
    "temporary_base_checkout",
    "validate_live_coverage_binding",
    "validate_terminal_coverage_binding",
    "validate_differential_coverage",
]
