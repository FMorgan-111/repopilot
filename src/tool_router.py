"""Safe execution of policy-approved local tool intents."""

from __future__ import annotations

import hashlib
import os
import stat
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Sequence

from pydantic import BaseModel

from .evidence import EvidenceCapacityError, EvidenceStore
from .local_search import (
    find_local_references,
    is_sensitive_repo_path,
    list_local_related_tests,
    read_local_range,
    read_local_symbol,
    search_local_symbol,
    search_local_text,
)
from .model_provider import redact_secrets
from .repo_paths import canonical_repo_path, canonical_repo_path_key
from .safe_subprocess import (
    BoundedProcessResult,
    SandboxPaths,
    run_bounded_process,
    run_oci_process,
    trusted_executable,
)
from .state import (
    AgentState,
    SnapshotManifestEntry,
    ToolAction,
    ToolInvocation,
    tool_manifest_fingerprint,
)
from .test_path_policy import is_allowed_test_path
from .tool_policy import ToolIntent, ToolPolicy


class ToolRouteResult(BaseModel):
    action: ToolAction
    status: Literal["rejected", "ok", "error", "duplicate"]
    args_fingerprint: str
    evidence_id: str | None = None
    error_class: str = ""
    made_progress: bool = False
    control_action: str = ""


_MAX_PATCH_BYTES = 2_000_000
_MAX_SNAPSHOT_FILES = 20_000
_MAX_SNAPSHOT_BYTES = 512_000_000
_MAX_ARCHIVE_BYTES = 768_000_000
_MAX_TEST_OVERLAY_FILES = 16
_MAX_TEST_OVERLAY_FILE_BYTES = 512_000
_MAX_TEST_OVERLAY_TOTAL_BYTES = 2_000_000
_PROTECTED_RUNTIME_MANIFESTS = frozenset(
    {
        "package.json",
        "package-lock.json",
        "npm-shrinkwrap.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lock",
        "bun.lockb",
        "pyproject.toml",
        "poetry.lock",
        "pdm.lock",
        "uv.lock",
    }
)


def _completed_content(result: BoundedProcessResult) -> str:
    return redact_secrets(
        f"returncode: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )[:8_000]


def _run(
    argv: list[str],
    root: Path,
    *,
    input_text: str | None = None,
    decode_errors: Literal["strict", "replace"] = "replace",
) -> BoundedProcessResult:
    return run_bounded_process(
        argv,
        cwd=root,
        input_text=input_text,
        timeout=300,
        max_output_bytes=8_000,
        decode_errors=decode_errors,
    )


def _approved_patch(state: AgentState) -> str:
    approval = state.tool_patch_approval
    if approval is None:
        return ""
    if approval.base_ref.lower() != state.repo_ref.lower():
        raise ValueError("approved patch base mismatch")
    patch_bytes = state.patch_content.encode("utf-8")
    if len(patch_bytes) > _MAX_PATCH_BYTES:
        raise ValueError("approved patch exceeds byte limit")
    normalized_patch = state.patch_content.replace("\r\n", "\n")
    if "\0" in state.patch_content or any(
        marker in normalized_patch
        for marker in ("GIT binary patch", "Binary files ")
    ):
        raise ValueError("binary patches are not allowed in tool snapshots")
    fingerprint = hashlib.sha256(state.patch_content.encode("utf-8")).hexdigest()
    if fingerprint != approval.patch_sha256:
        raise ValueError("approved patch content mismatch")
    if tool_manifest_fingerprint(approval.changed_manifest) != approval.manifest_fingerprint:
        raise ValueError("approved patch manifest fingerprint mismatch")
    for entry in approval.changed_manifest:
        if is_sensitive_repo_path(entry.path):
            raise ValueError("approved manifest contains a sensitive path")
    return state.patch_content


def _preflight_exact_tree(root: Path, ref: str) -> tuple[int, int]:
    """Count the exact tree and all blob bytes before archive construction."""
    git = trusted_executable("git")
    result = run_bounded_process(
        [git, "-C", str(root), "ls-tree", "-r", "-l", "-z", ref],
        cwd=root,
        timeout=60,
        max_output_bytes=8_000_000,
        decode_errors="strict",
    )
    if result.returncode != 0:
        raise RuntimeError("exact tree preflight failed")
    count = 0
    total = 0
    folded_paths: set[str] = set()
    for record in (item for item in result.stdout.split("\0") if item):
        metadata, separator, path = record.partition("\t")
        fields = metadata.split()
        if not separator or len(fields) != 4 or fields[1] != "blob":
            raise ValueError("invalid exact tree record")
        mode, _kind, _object_id, raw_size = fields
        if mode not in {"100644", "100755"}:
            raise ValueError("exact tree contains a symlink or special entry")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ValueError("invalid exact tree blob size") from exc
        normalized = canonical_repo_path(path)
        folded = canonical_repo_path_key(normalized)
        if (
            not normalized
            or Path(normalized).is_absolute()
            or ".." in Path(normalized).parts
            or folded in folded_paths
        ):
            raise ValueError("exact tree contains an unsafe or colliding path")
        folded_paths.add(folded)
        count += 1
        total += size
        if count > _MAX_SNAPSHOT_FILES:
            raise ValueError("exact tree file count exceeds limit")
        if total > _MAX_SNAPSHOT_BYTES:
            raise ValueError("exact tree blob bytes exceed limit")
    return count, total


def _archive_byte_limit(file_count: int, blob_bytes: int) -> int:
    derived = blob_bytes + file_count * 8_192 + 1_048_576
    return min(_MAX_ARCHIVE_BYTES, derived)


def _base_tree_has_sensitive_paths(root: Path, ref: str) -> bool:
    result = run_bounded_process(
        [trusted_executable("git"), "-C", str(root), "ls-tree", "-r", "--name-only", "-z", ref],
        cwd=root,
        timeout=60,
        max_output_bytes=8_000_000,
        decode_errors="strict",
    )
    if result.returncode != 0:
        raise RuntimeError("sensitive base path preflight failed")
    seen: set[str] = set()
    sensitive = False
    for raw_path in (item for item in result.stdout.split("\0") if item):
        path = canonical_repo_path(raw_path)
        key = canonical_repo_path_key(path)
        if key in seen:
            raise ValueError("base tree contains a canonical path collision")
        seen.add(key)
        sensitive = sensitive or is_sensitive_repo_path(path)
    return sensitive


def _extract_safe_archive(archive: Path, workspace: Path) -> None:
    total_size = 0
    file_count = 0
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:") as bundle:
        for member in bundle:
            raw = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
            raw = canonical_repo_path(raw)
            parts = Path(raw).parts
            if not raw or Path(raw).is_absolute() or ".." in parts:
                raise ValueError("unsafe archive path")
            if is_sensitive_repo_path(raw):
                continue
            folded = canonical_repo_path_key(raw)
            if folded in seen:
                raise ValueError("archive contains a colliding path")
            seen.add(folded)
            destination = (workspace / raw).resolve()
            if not destination.is_relative_to(workspace):
                raise ValueError("archive path escaped snapshot")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o777)
                continue
            if not member.isfile():
                raise ValueError("archive contains a symlink or special entry")
            file_count += 1
            total_size += member.size
            if file_count > _MAX_SNAPSHOT_FILES:
                raise ValueError("repository snapshot file count exceeds limit")
            if total_size > _MAX_SNAPSHOT_BYTES:
                raise ValueError("repository snapshot too large")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("archive member missing content")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                while chunk := source.read(64 * 1024):
                    output.write(chunk)
            destination.chmod(0o777 if member.mode & 0o111 else 0o666)


def _scan_snapshot(workspace: Path) -> dict[str, tuple[str, str, int]]:
    """Scan without following links and hash every regular snapshot file."""
    manifest: dict[str, tuple[str, str, int]] = {}
    folded_paths: set[str] = set()
    stack = [workspace]
    total = 0
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(workspace).as_posix()
                relative = canonical_repo_path(relative)
                folded = canonical_repo_path_key(relative)
                if folded in folded_paths:
                    raise ValueError("snapshot contains a case-colliding path")
                folded_paths.add(folded)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError("snapshot contains a symlink")
                if stat.S_ISDIR(info.st_mode):
                    stack.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("snapshot contains a special file")
                if is_sensitive_repo_path(relative):
                    raise ValueError("snapshot contains a sensitive path")
                if len(manifest) >= _MAX_SNAPSHOT_FILES:
                    raise ValueError("snapshot file count exceeds limit")
                total += info.st_size
                if total > _MAX_SNAPSHOT_BYTES:
                    raise ValueError("snapshot bytes exceed limit")
                mode = "100755" if info.st_mode & 0o111 else "100644"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest[relative] = (mode, digest, info.st_size)
    return dict(sorted(manifest.items()))


def _changed_manifest(
    base: dict[str, tuple[str, str, int]],
    current: dict[str, tuple[str, str, int]],
) -> list[SnapshotManifestEntry]:
    changed: list[SnapshotManifestEntry] = []
    for path in sorted(set(base) | set(current)):
        before = base.get(path)
        after = current.get(path)
        if before == after:
            continue
        if Path(path).name.casefold() in _PROTECTED_RUNTIME_MANIFESTS:
            raise ValueError("approved patch changes a protected runtime manifest")
        if is_sensitive_repo_path(path):
            raise ValueError("approved patch changes a sensitive path")
        if after is None:
            assert before is not None
            changed.append(
                SnapshotManifestEntry(
                    path=path,
                    change="deleted",
                    mode=before[0],
                    content_sha256=None,
                    size=before[2],
                )
            )
        else:
            changed.append(
                SnapshotManifestEntry(
                    path=path,
                    change="added" if before is None else "modified",
                    mode=after[0],
                    content_sha256=after[1],
                    size=after[2],
                )
            )
    return changed


def _preflight_approved_manifest(
    base: dict[str, tuple[str, str, int]],
    entries: list[SnapshotManifestEntry],
    patch_bytes: int,
) -> None:
    predicted_count = len(base)
    predicted_bytes = sum(item[2] for item in base.values())
    for entry in entries:
        before = base.get(entry.path)
        if entry.change == "added":
            if before is not None or entry.size > patch_bytes:
                raise ValueError("approved added entry has an impossible size")
            predicted_count += 1
        else:
            if before is None:
                raise ValueError("approved changed entry is absent from base")
            predicted_bytes -= before[2]
            if entry.change == "deleted":
                if entry.size != before[2]:
                    raise ValueError("approved deletion size does not match base")
                predicted_count -= 1
            elif entry.size > before[2] + patch_bytes:
                raise ValueError("approved modified entry has an impossible size")
        if entry.change != "deleted":
            predicted_bytes += entry.size
        if predicted_count > _MAX_SNAPSHOT_FILES or predicted_count < 0:
            raise ValueError("approved post-patch file count exceeds limit")
        if predicted_bytes > _MAX_SNAPSHOT_BYTES or predicted_bytes < 0:
            raise ValueError("approved post-patch bytes exceed limit")


def _validate_generated_snapshot(state: AgentState, workspace: Path) -> None:
    approval = state.tool_patch_approval
    seen: set[str] = set()
    for generated in state.generated_test_approvals:
        if approval is None or generated.patch_gate_fingerprint != approval.patch_gate_fingerprint:
            raise ValueError("generated test lacks matching PatchGate approval")
        if generated.path in seen or is_sensitive_repo_path(generated.path):
            raise ValueError("generated test approvals are not unique and safe")
        seen.add(generated.path)
        path = (workspace / generated.path).resolve()
        source = (Path(state.repo_path).resolve() / generated.path).resolve()
        if (
            not path.is_relative_to(workspace)
            or not source.is_relative_to(Path(state.repo_path).resolve())
            or path.is_symlink()
            or source.is_symlink()
            or not path.is_file()
            or not source.is_file()
        ):
            raise ValueError("approved generated test missing from snapshot")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_entry = next(
            (entry for entry in approval.changed_manifest if entry.path == generated.path),
            None,
        )
        if (
            digest != generated.content_sha256
            or source_digest != generated.content_sha256
            or manifest_entry is None
            or manifest_entry.change != generated.change
            or manifest_entry.content_sha256 != generated.content_sha256
        ):
            raise ValueError("approved generated test content mismatch")


def _read_overlay_source(root: Path, relative: str) -> bytes:
    current = root
    for component in Path(relative).parts:
        current = current / component
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("generated test overlay crosses a symlink")
    info = current.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_TEST_OVERLAY_FILE_BYTES:
        raise ValueError("generated test overlay is not a bounded regular file")
    descriptor = os.open(current, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (info.st_dev, info.st_ino):
            raise ValueError("generated test overlay identity drifted")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 65_536):
            total += len(chunk)
            if total > _MAX_TEST_OVERLAY_FILE_BYTES:
                raise ValueError("generated test overlay exceeds its byte cap")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    text = content.decode("utf-8")
    if "\0" in text:
        raise ValueError("generated test overlay contains NUL")
    return content


def _approved_test_overlays(
    state: AgentState,
    paths: Sequence[str] | None,
) -> tuple[tuple[str, bytes], ...]:
    if paths is None:
        return ()
    if len(paths) > _MAX_TEST_OVERLAY_FILES:
        raise ValueError("too many generated test overlays")
    approval = state.tool_patch_approval
    if approval is None:
        raise ValueError("generated test approval requires a combined patch approval")
    root = Path(state.repo_path).resolve(strict=True)
    normalized = [canonical_repo_path(path) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError("generated test overlay paths must be unique")
    generated = {item.path: item for item in state.generated_test_approvals}
    entries = {item.path: item for item in approval.changed_manifest}
    overlays: list[tuple[str, bytes]] = []
    total = 0
    for path in normalized:
        if (
            path not in state.coverage_test_files
            or is_sensitive_repo_path(path)
            or not is_allowed_test_path(root, path)
        ):
            raise ValueError("generated test overlay is not an approved test path")
        marker = generated.get(path)
        entry = entries.get(path)
        if marker is None:
            raise ValueError("generated test approval is missing")
        if (
            marker.patch_gate_fingerprint != approval.patch_gate_fingerprint
            or entry is None
            or entry.change not in {"added", "modified"}
            or marker.change != entry.change
            or marker.content_sha256 != entry.content_sha256
        ):
            raise ValueError("generated test approval does not match the combined manifest")
        content = _read_overlay_source(root, path)
        digest = hashlib.sha256(content).hexdigest()
        if digest != marker.content_sha256 or len(content) != entry.size:
            raise ValueError("generated test overlay digest does not match approval")
        base = run_bounded_process(
            [trusted_executable("git"), "-C", str(root), "show", f"{state.repo_ref}:{path}"],
            cwd=root,
            timeout=30,
            max_output_bytes=_MAX_TEST_OVERLAY_FILE_BYTES + 1,
            decode_errors="strict",
        )
        if marker.change == "added":
            if base.returncode == 0:
                raise ValueError("added generated test unexpectedly exists in base")
        elif (
            base.returncode != 0
            or marker.base_content_sha256
            != hashlib.sha256(base.stdout.encode("utf-8")).hexdigest()
        ):
            raise ValueError("modified generated test base preimage mismatch")
        total += len(content)
        if total > _MAX_TEST_OVERLAY_TOTAL_BYTES:
            raise ValueError("generated test overlays exceed total byte cap")
        overlays.append((path, content))
    return tuple(overlays)


@contextmanager
def disposable_test_snapshot(
    state: AgentState,
    *,
    apply_approved_changes: bool = True,
    test_overlay_paths: Sequence[str] | None = None,
) -> Iterator[SandboxPaths]:
    """Yield a fresh exact-base archive snapshot, optionally with its approved diff."""
    with tempfile.TemporaryDirectory(prefix="repopilot-tool-sandbox-") as directory:
        sandbox = SandboxPaths.create(Path(directory) / "private")
        archive = sandbox.root / "base.tar"
        git = trusted_executable("git")
        file_count, blob_bytes = _preflight_exact_tree(
            Path(state.repo_path).resolve(), state.repo_ref
        )
        archive_limit = _archive_byte_limit(file_count, blob_bytes)
        archived = run_bounded_process(
            [git, "-C", state.repo_path, "archive", "--format=tar", f"--output={archive}", state.repo_ref],
            cwd=state.repo_path,
            timeout=60,
            max_output_bytes=8_000,
            watched_output_path=archive,
            max_watched_output_bytes=archive_limit,
        )
        if (
            archived.returncode != 0
            or not archive.is_file()
            or archive.stat().st_size > archive_limit
        ):
            raise RuntimeError("exact-base snapshot creation failed")
        _extract_safe_archive(archive, sandbox.workspace)
        base_manifest = _scan_snapshot(sandbox.workspace)
        patch = _approved_patch(state) if apply_approved_changes else ""
        if apply_approved_changes and state.tool_patch_approval is not None:
            _preflight_approved_manifest(
                base_manifest,
                state.tool_patch_approval.changed_manifest,
                len(patch.encode("utf-8")),
            )
        if patch:
            for check_only in (True, False):
                argv = [git, "apply", "--recount"]
                if check_only:
                    argv.append("--check")
                argv.append("-")
                result = run_bounded_process(
                    argv,
                    cwd=sandbox.workspace,
                    input_text=patch,
                    timeout=60,
                    max_output_bytes=8_000,
                )
                if result.returncode != 0:
                    raise ValueError("approved patch does not apply to exact base snapshot")
        current_manifest = _scan_snapshot(sandbox.workspace)
        actual_changes = _changed_manifest(base_manifest, current_manifest)
        expected_changes = (
            state.tool_patch_approval.changed_manifest
            if apply_approved_changes and state.tool_patch_approval is not None
            else []
        )
        if tuple(actual_changes) != tuple(expected_changes):
            raise ValueError("snapshot changes do not match approved manifest")
        if apply_approved_changes:
            _validate_generated_snapshot(state, sandbox.workspace)
        for raw_path, content in _approved_test_overlays(state, test_overlay_paths):
            path = canonical_repo_path(raw_path)
            if (
                is_sensitive_repo_path(path)
                or len(content) > 512_000
            ):
                raise ValueError("unsafe disposable test overlay")
            destination = sandbox.workspace / path
            current = sandbox.workspace
            for part in Path(path).parts[:-1]:
                current = current / part
                if current.is_symlink():
                    raise ValueError("disposable test overlay crosses a symlink")
                current.mkdir(exist_ok=True)
            if destination.is_symlink() or (
                destination.exists() and not destination.is_file()
            ):
                raise ValueError("disposable test overlay target is unsafe")
            resolved = destination.resolve(strict=False)
            if not resolved.is_relative_to(sandbox.workspace.resolve()):
                raise ValueError("disposable test overlay escaped workspace")
            destination.write_bytes(content)
        locked_manifest = _scan_snapshot(sandbox.workspace)
        yield sandbox
        if _scan_snapshot(sandbox.workspace) != locked_manifest:
            raise ValueError("disposable test snapshot manifest drifted during execution")


@contextmanager
def _disposable_test_snapshot(state: AgentState) -> Iterator[SandboxPaths]:
    """Backward-compatible private wrapper for existing targeted-test routing."""
    with disposable_test_snapshot(state, apply_approved_changes=True) as sandbox:
        yield sandbox


def _changed_path_groups(state: AgentState, root: Path) -> list[tuple[str, ...]]:
    git = trusted_executable("git")
    result = _run(
        [
            git,
            "-C",
            str(root),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            "--find-renames",
            state.repo_ref,
            "--",
        ],
        root,
        decode_errors="strict",
    )
    if result.returncode != 0:
        raise RuntimeError("changed path enumeration failed")
    fields = [field for field in result.stdout.split("\0") if field]
    groups: list[tuple[str, ...]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        count = 2 if status[:1] in {"R", "C"} else 1
        if index + count > len(fields):
            raise ValueError("invalid changed path record")
        paths = tuple(fields[index : index + count])
        index += count
        groups.append(paths)
    return groups


def _safe_changed_paths(state: AgentState, root: Path) -> tuple[list[str], bool]:
    accepted: list[str] = []
    sensitive_changed = False
    for raw_group in _changed_path_groups(state, root):
        group = tuple(canonical_repo_path(path) for path in raw_group)
        if any(is_sensitive_repo_path(path) for path in group):
            sensitive_changed = True
            continue
        for path in group:
            if path not in accepted:
                accepted.append(path)
    return accepted, sensitive_changed


def _has_untracked_files(root: Path) -> bool:
    result = _run(
        [
            trusted_executable("git"),
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        root,
        decode_errors="strict",
    )
    if result.returncode != 0:
        raise RuntimeError("untracked file enumeration failed")
    return bool(result.stdout)


def _execute_data_tool(
    state: AgentState,
    intent: ToolIntent,
    normalized_args: dict[str, object],
    argv: list[str],
) -> tuple[str, str | None, str | None]:
    root = Path(state.repo_path).resolve()
    action = intent.action
    path = normalized_args.get("path")
    file_path = str(path) if isinstance(path, str) else None
    symbol_arg = normalized_args.get("symbol")
    symbol = str(symbol_arg) if isinstance(symbol_arg, str) else None
    if action == "search_symbol":
        content = search_local_symbol(state.repo_path, symbol or "", file_path or "")
    elif action == "search_text":
        content = search_local_text(
            state.repo_path, str(normalized_args["text"]), file_path or ""
        )
    elif action == "read_symbol":
        content = read_local_symbol(state.repo_path, file_path or "", symbol or "")
    elif action == "read_range":
        content = read_local_range(
            state.repo_path,
            file_path or "",
            int(normalized_args["start_line"]),
            int(normalized_args["end_line"]),
        )
    elif action == "find_references":
        content = find_local_references(state.repo_path, symbol or "", file_path or "")
    elif action == "list_related_tests":
        content = "\n".join(list_local_related_tests(state.repo_path, file_path or ""))
    elif action == "run_targeted_test":
        with _disposable_test_snapshot(state) as sandbox:
            if state.tool_sandbox_config is None:
                raise ValueError("tool sandbox is not configured")
            content = _completed_content(
                run_oci_process(
                    argv,
                    sandbox=sandbox,
                    config=state.tool_sandbox_config,
                    timeout=300,
                    max_output_bytes=8_000,
                )
            )
    elif action == "inspect_git_diff":
        base_has_sensitive = _base_tree_has_sensitive_paths(root, state.repo_ref)
        safe_paths, sensitive_changed = _safe_changed_paths(state, root)
        if base_has_sensitive or sensitive_changed:
            content = (
                "Diff evidence omitted because a sensitive path exists in or changed "
                "from the exact base tree."
            )
        elif not safe_paths:
            content = "No non-sensitive tracked changes."
        else:
            content = _completed_content(
                _run(
                    [
                        trusted_executable("git"),
                        "diff",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--unified=3",
                        state.repo_ref,
                        "--",
                        *(f":(literal){path}" for path in safe_paths),
                    ],
                    root,
                )
            )
        if _has_untracked_files(root):
            content += "\nUntracked files are conservatively omitted from diff evidence."
    elif action == "validate_patch":
        result = _run(
            [trusted_executable("git"), "apply", "--check", "--recount", "-"],
            root,
            input_text=state.patch_content,
        )
        if result.returncode != 0:
            result = _run(
                [
                    trusted_executable("git"),
                    "apply",
                    "--check",
                    "--reverse",
                    "--recount",
                    "-",
                ],
                root,
                input_text=state.patch_content,
            )
        content = (
            "Patch valid for the exact checkout."
            if result.returncode == 0
            else _completed_content(result)
        )
    else:  # pragma: no cover - policy and caller keep this unreachable
        raise ValueError("unsupported data tool")
    return redact_secrets(content)[:8_000], file_path, symbol


async def route_tool_intent(
    state: AgentState,
    intent: ToolIntent,
    *,
    calls_this_round: int,
) -> ToolRouteResult:
    decision = ToolPolicy().authorize(
        state, intent, calls_this_round=calls_this_round
    )
    if not decision.approved:
        invocation = ToolInvocation(
            action=intent.action,
            args_fingerprint=decision.args_fingerprint,
            status=decision.status,
        )
        state.tool_history.append(invocation)
        return ToolRouteResult(
            action=intent.action,
            status=decision.status,
            args_fingerprint=decision.args_fingerprint,
        )

    if intent.action in {"request_repair", "finish_investigation"}:
        state.tool_history.append(
            ToolInvocation(
                action=intent.action,
                args_fingerprint=decision.args_fingerprint,
                status="ok",
            )
        )
        return ToolRouteResult(
            action=intent.action,
            status="ok",
            args_fingerprint=decision.args_fingerprint,
            control_action=intent.action,
        )

    try:
        content, file_path, symbol = _execute_data_tool(
            state, intent, decision.normalized_args, decision.argv
        )
        added = EvidenceStore(state).add(
            tool=intent.action,
            summary=f"{intent.action} result",
            content=content,
            file_path=file_path,
            symbol=symbol,
        )
        if added.disposition == "capacity":
            raise EvidenceCapacityError("evidence store reached its item limit")
        if added.evidence is None:  # pragma: no cover - model contract guard
            raise EvidenceCapacityError("evidence result omitted its persisted item")
        status = "ok" if added.disposition == "added" else "duplicate"
        state.tool_history.append(
            ToolInvocation(
                action=intent.action,
                args_fingerprint=decision.args_fingerprint,
                status=status,
                evidence_id=added.evidence.evidence_id,
            )
        )
        return ToolRouteResult(
            action=intent.action,
            status=status,
            args_fingerprint=decision.args_fingerprint,
            evidence_id=added.evidence.evidence_id,
            made_progress=added.added,
        )
    except Exception as exc:
        error_class = type(exc).__name__
        state.tool_history.append(
            ToolInvocation(
                action=intent.action,
                args_fingerprint=decision.args_fingerprint,
                status="error",
                error_class=error_class,
            )
        )
        return ToolRouteResult(
            action=intent.action,
            status="error",
            args_fingerprint=decision.args_fingerprint,
            error_class=error_class,
        )
