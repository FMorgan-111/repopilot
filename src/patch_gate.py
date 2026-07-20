"""Fail-closed, non-mutating validation for model-authored verified edits."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .local_search import is_sensitive_repo_path
from .model_provider import redact_secrets
from .patch_match import closest_region, leading_spaces, locate_node_span, reindent
from .repo_paths import canonical_repo_path
from .repair_flow import RepairContextError, _read_regular_no_follow
from .state import (
    AgentState,
    PatchEdit,
    RepairPlan,
    SnapshotManifestEntry,
    ToolPatchApproval,
    VerifiedEditBatch,
    tool_manifest_fingerprint,
)

MAX_PATCH_FILE_BYTES = 512_000
MAX_CORRECTION_CONTEXT_CHARS = 1_200

_ARCHIVE_SUFFIXES = frozenset(
    {".7z", ".bz2", ".egg", ".gz", ".rar", ".tar", ".tgz", ".whl", ".xz", ".zip"}
)
_BINARY_SUFFIXES = frozenset(
    {
        ".a", ".avi", ".bin", ".class", ".dll", ".dylib", ".exe", ".gif",
        ".ico", ".jar", ".jpeg", ".jpg", ".mov", ".mp3", ".mp4", ".o",
        ".pdf", ".png", ".pyc", ".pyo", ".so", ".wasm", ".webp",
    }
)
_GENERATED_PARTS = frozenset(
    {
        ".cache", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache",
        ".tox", ".venv", "__pycache__", "build", "coverage", "dist",
        "htmlcov", "node_modules", "site-packages", "target", "venv",
    }
)
_NEW_TEXT_SUFFIXES = frozenset(
    {
        ".c", ".cc", ".cfg", ".conf", ".cpp", ".css", ".go", ".h",
        ".hpp", ".html", ".ini", ".java", ".js", ".json", ".jsx", ".md",
        ".php", ".py", ".rb", ".rs", ".rst", ".sh", ".toml", ".ts",
        ".tsx", ".txt", ".xml", ".yaml", ".yml",
    }
)


class PatchGateIssue(BaseModel):
    code: Literal[
        "target_missing",
        "target_ambiguous",
        "search_missing",
        "scope_violation",
        "apply_failed",
        "empty_patch",
        "generated_artifact",
        "binary_artifact",
        "oversized_file",
    ]
    file_path: str
    message: str
    correction_context: str = ""


class PatchGateResult(BaseModel):
    accepted: bool
    edits: list[PatchEdit] = Field(default_factory=list)
    issues: list[PatchGateIssue] = Field(default_factory=list)


def _clear_gate_output(state: AgentState) -> None:
    """Retire every value transitively bound to a rejected patch batch."""
    state.patch_content = ""
    state.patch_edits = []
    state.tool_patch_approval = None
    state.generated_test_approvals = []


def _issue(
    code: str,
    path: str,
    message: str,
    content: str = "",
) -> PatchGateIssue:
    return PatchGateIssue(
        code=code,  # type: ignore[arg-type]
        file_path=redact_secrets(path)[:500],
        message=redact_secrets(message)[:500],
        correction_context=redact_secrets(content)[:MAX_CORRECTION_CONTEXT_CHARS],
    )


def _generated_or_binary_issue(path: str) -> PatchGateIssue | None:
    lowered = path.casefold()
    suffix = Path(lowered).suffix
    parts = set(Path(lowered).parts)
    if suffix in _ARCHIVE_SUFFIXES or any(
        part in _GENERATED_PARTS or part.endswith((".egg-info", ".dist-info"))
        for part in parts
    ):
        return _issue("generated_artifact", path, "Generated, packaged, or cached artifacts are not editable.")
    if suffix in _BINARY_SUFFIXES:
        return _issue("binary_artifact", path, "Binary artifacts are not editable.")
    return None


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        timeout=30,
    )


def _exact_root(state: AgentState) -> Path:
    if not state.repo_path or not state.repo_ref or len(state.repo_ref) != 40:
        raise ValueError("PatchGate requires an exact base checkout")
    root = Path(state.repo_path).resolve()
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode or head.stdout.decode("ascii", "ignore").strip().lower() != state.repo_ref.lower():
        raise ValueError("PatchGate checkout does not match the exact base")
    return root


def _tree_entry(root: Path, ref: str, path: str) -> tuple[str, bytes] | None:
    listed = _git(root, "ls-tree", "-z", ref, "--", path)
    if listed.returncode or not listed.stdout:
        return None
    records = [item for item in listed.stdout.split(b"\0") if item]
    if len(records) != 1:
        return None
    metadata, separator, found = records[0].partition(b"\t")
    fields = metadata.split()
    if not separator or found.decode("utf-8", "strict") != path or len(fields) != 3:
        return None
    mode = fields[0].decode("ascii")
    if mode not in {"100644", "100755"} or fields[1] != b"blob":
        raise ValueError("tracked target is a symlink or special file")
    blob = _git(root, "show", f"{ref}:{path}")
    if blob.returncode:
        raise ValueError("tracked target preimage is unavailable")
    return mode, blob.stdout


def _safe_leaf(root: Path, path: str, *, must_exist: bool) -> Path:
    current = root
    parts = path.split("/")
    for index, component in enumerate(parts):
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if must_exist or index != len(parts) - 1:
                # Missing parent directories are safe for an intentional new file.
                if must_exist:
                    raise ValueError("target file is missing")
                return current
            return current
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("target path contains a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ValueError("target parent is not a directory")
        if index == len(parts) - 1:
            if must_exist and not stat.S_ISREG(info.st_mode):
                raise ValueError("target is not a regular file")
            if not must_exist:
                raise ValueError("intentional new target already exists")
    return current


def _decode_text(data: bytes) -> str:
    text = data.decode("utf-8")
    if "\0" in text:
        raise UnicodeDecodeError("utf-8", data, 0, 1, "NUL byte")
    return text


def _normalize_with_boundaries(text: str) -> tuple[str, list[int]]:
    """Normalize newlines while retaining exact raw offsets for replacements."""
    normalized: list[str] = []
    boundaries = [0]
    index = 0
    while index < len(text):
        if text[index] == "\r":
            index += 2 if index + 1 < len(text) and text[index + 1] == "\n" else 1
            normalized.append("\n")
        else:
            normalized.append(text[index])
            index += 1
        boundaries.append(index)
    return "".join(normalized), boundaries


def _newline_style(text: str, start: int = 0, end: int | None = None) -> str:
    sample = text[start:end]
    crlf = sample.count("\r\n")
    bare_cr = sample.count("\r") - crlf
    bare_lf = sample.count("\n") - crlf
    if crlf or bare_cr or bare_lf:
        return max(((crlf, "\r\n"), (bare_cr, "\r"), (bare_lf, "\n")))[1]
    crlf = text.count("\r\n")
    bare_cr = text.count("\r") - crlf
    bare_lf = text.count("\n") - crlf
    if not (crlf or bare_cr or bare_lf):
        return "\n"
    return max(((crlf, "\r\n"), (bare_cr, "\r"), (bare_lf, "\n")))[1]


def _raw_replacement(replacement: str, newline: str) -> str:
    logical, _ = _normalize_with_boundaries(replacement)
    return logical.replace("\n", newline)


def _node_replace(content: str, target: str, replacement: str) -> str | None:
    logical, boundaries = _normalize_with_boundaries(content)
    span = locate_node_span(logical, target)
    if span is None:
        return None
    start, end, indent = span
    first = next((line for line in replacement.split("\n") if line.strip()), "")
    body = reindent(replacement, indent - leading_spaces(first))
    if not body.endswith("\n"):
        body += "\n"
    raw_start, raw_end = boundaries[start], boundaries[end]
    body = _raw_replacement(body, _newline_style(content, raw_start, raw_end))
    return content[:raw_start] + body + content[raw_end:]


def _node_target_count(content: str, target: str) -> int:
    content, _ = _normalize_with_boundaries(content)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0
    count = 0

    def walk(node: ast.AST, stack: list[str]) -> None:
        nonlocal count
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [*stack, child.name]
                if ".".join(names) == target:
                    count += 1
                walk(child, names)
            else:
                walk(child, stack)

    walk(tree, [])
    return count


def _patch_text(
    before: dict[str, str | None],
    after: dict[str, str],
    modes: dict[str, str],
) -> str:
    """Ask Git to render raw-byte-safe patches, including CR-only files."""
    chunks: list[bytes] = []
    with tempfile.TemporaryDirectory(prefix="repopilot-patch-") as temporary:
        scratch = Path(temporary)
        for path in sorted(after):
            old = before[path]
            if old == after[path]:
                continue
            new_path = scratch / "b" / path
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_bytes(after[path].encode("utf-8"))
            new_path.chmod(0o755 if modes[path] == "100755" else 0o644)
            if old is None:
                old_arg = "/dev/null"
            else:
                old_path = scratch / "a" / path
                old_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.write_bytes(old.encode("utf-8"))
                old_path.chmod(0o755 if modes[path] == "100755" else 0o644)
                old_arg = f"a/{path}"
            result = subprocess.run(
                [
                    "git", "diff", "--no-index", "--binary", "--no-ext-diff",
                    "--src-prefix=", "--dst-prefix=", "--", old_arg, f"b/{path}",
                ],
                cwd=scratch,
                check=False,
                capture_output=True,
                timeout=30,
            )
            if result.returncode not in {0, 1}:
                raise ValueError("Git could not render the exact approved patch")
            chunk = result.stdout
            if old is None:
                chunk = chunk.replace(
                    f"diff --git b/{path} b/{path}\n".encode(),
                    f"diff --git a/{path} b/{path}\n".encode(),
                    1,
                )
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def _fingerprint(
    state: AgentState,
    plan: RepairPlan,
    edits: list[PatchEdit],
    manifest: list[SnapshotManifestEntry],
    patch_sha256: str,
) -> str:
    payload = {
        "base_ref": state.repo_ref.lower(),
        "plan": plan.model_dump(mode="json"),
        "ordered_exact_edits": [edit.model_dump(mode="json") for edit in edits],
        "result_manifest": [entry.model_dump(mode="json") for entry in manifest],
        "patch_sha256": patch_sha256,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_patch_batch(
    state: AgentState,
    plan: RepairPlan,
    batch: VerifiedEditBatch,
) -> PatchGateResult:
    """Simulate a complete edit batch and authorize its exact resulting diff."""
    issues: list[PatchGateIssue] = []
    try:
        root = _exact_root(state)
    except ValueError as exc:
        _clear_gate_output(state)
        return PatchGateResult(accepted=False, issues=[_issue("apply_failed", "", str(exc))])
    if state.active_repair_plan is None or state.active_repair_plan != plan:
        _clear_gate_output(state)
        return PatchGateResult(
            accepted=False,
            issues=[_issue("scope_violation", "", "RepairPlan is not the active exact plan.")],
        )

    before: dict[str, str | None] = {}
    after: dict[str, str] = {}
    modes: dict[str, str] = {}
    converted: list[PatchEdit] = []
    seen_anchors: set[tuple[str, str, str]] = set()
    for edit in batch.edits:
        raw_path = edit.file_path
        try:
            path = canonical_repo_path(raw_path)
        except ValueError:
            issues.append(_issue("scope_violation", str(raw_path), "Target path is not canonical and repository-relative."))
            continue
        artifact_issue = _generated_or_binary_issue(path)
        if artifact_issue is not None:
            issues.append(artifact_issue)
            continue
        if is_sensitive_repo_path(path):
            issues.append(_issue("scope_violation", path, "Sensitive repository paths are outside patch scope."))
            continue
        if path not in plan.target_files:
            issues.append(_issue("scope_violation", path, "Edit target is outside the active RepairPlan."))
            continue
        anchor = (path, edit.node_target or "", edit.search)
        if anchor in seen_anchors:
            issues.append(_issue("target_ambiguous", path, "Verified edit anchors must be unique."))
            continue
        seen_anchors.add(anchor)

        try:
            tracked = _tree_entry(root, state.repo_ref, path)
        except (UnicodeDecodeError, ValueError) as exc:
            issues.append(_issue("scope_violation", path, str(exc)))
            continue
        if tracked is None:
            suffix = Path(path).suffix.casefold()
            if suffix not in _NEW_TEXT_SUFFIXES:
                issues.append(_issue("target_missing", path, "Target is neither tracked nor an intentional new text file."))
                continue
            try:
                _safe_leaf(root, path, must_exist=False)
            except ValueError as exc:
                issues.append(_issue("scope_violation", path, str(exc)))
                continue
            if edit.search or edit.node_target:
                issues.append(_issue("target_missing", path, "Intentional new files cannot use existing anchors."))
                continue
            if path in before:
                issues.append(_issue("target_ambiguous", path, "Intentional new files may be created only once."))
                continue
            if not edit.replace.strip():
                issues.append(_issue("empty_patch", path, "Intentional new file content is empty."))
                continue
            encoded = edit.replace.encode("utf-8")
            if len(encoded) > MAX_PATCH_FILE_BYTES:
                issues.append(_issue("oversized_file", path, "Resulting file exceeds the PatchGate size limit."))
                continue
            if "\0" in edit.replace:
                issues.append(_issue("binary_artifact", path, "Intentional new file is not text."))
                continue
            empty_digest = hashlib.sha256(b"").hexdigest()
            if edit._expected_content_sha256 and edit._expected_content_sha256 != empty_digest:
                issues.append(_issue("apply_failed", path, "Verified edit is not bound to the empty new-file preimage."))
                continue
            edit._expected_content_sha256 = empty_digest
            edit._exact_only = True
            before[path] = None
            after[path] = edit.replace
            modes[path] = "100644"
            converted.append(
                PatchEdit(
                    file_path=path,
                    replace=edit.replace,
                    expected_content_sha256=empty_digest,
                    exact_only=True,
                )
            )
            continue

        mode, base_bytes = tracked
        try:
            _safe_leaf(root, path, must_exist=True)
            live_bytes = _read_regular_no_follow(root, path)
            live_source = _decode_text(live_bytes)
            base_source = _decode_text(base_bytes)
        except (OSError, RepairContextError, UnicodeDecodeError, ValueError):
            issues.append(_issue("binary_artifact", path, "Tracked target is not stable UTF-8 regular text."))
            continue
        if len(live_bytes) > MAX_PATCH_FILE_BYTES:
            issues.append(_issue("oversized_file", path, "Target file exceeds the PatchGate size limit."))
            continue
        expected = hashlib.sha256(live_bytes).hexdigest()
        if live_bytes != base_bytes or (
            edit._expected_content_sha256
            and edit._expected_content_sha256 != expected
        ):
            issues.append(_issue("apply_failed", path, "Target changed from the exact verified preimage.", live_source))
            continue
        edit._expected_content_sha256 = expected
        edit._exact_only = True
        if live_source != base_source:
            issues.append(_issue("apply_failed", path, "Target text does not match the exact base.", live_source))
            continue
        source = after.get(path, live_source)
        if edit.node_target:
            if edit.node_target not in plan.target_symbols:
                issues.append(_issue("scope_violation", path, "Node target is outside the RepairPlan symbols.", source))
                continue
            updated = _node_replace(source, edit.node_target, edit.replace)
            if updated is None:
                count = _node_target_count(source, edit.node_target)
                code = "target_ambiguous" if count > 1 else "target_missing"
                window = closest_region(source, edit.node_target.rsplit(".", 1)[-1], max_chars=MAX_CORRECTION_CONTEXT_CHARS)
                issues.append(_issue(code, path, "Node target is missing or ambiguous.", window))
                continue
        else:
            logical_source, boundaries = _normalize_with_boundaries(source)
            logical_search, _ = _normalize_with_boundaries(edit.search)
            count = logical_source.count(logical_search)
            if count == 0:
                window = closest_region(logical_source, logical_search, max_chars=MAX_CORRECTION_CONTEXT_CHARS)
                issues.append(_issue("search_missing", path, "Exact search block is missing; fuzzy matching is disabled.", window))
                continue
            if count != 1:
                window = closest_region(logical_source, logical_search, max_chars=MAX_CORRECTION_CONTEXT_CHARS)
                issues.append(_issue("target_ambiguous", path, "Exact search block is not unique.", window))
                continue
            start = logical_source.index(logical_search)
            end = start + len(logical_search)
            raw_start, raw_end = boundaries[start], boundaries[end]
            replacement = _raw_replacement(
                edit.replace,
                _newline_style(source, raw_start, raw_end),
            )
            updated = source[:raw_start] + replacement + source[raw_end:]
        if updated == source:
            issues.append(_issue("empty_patch", path, "Edit produces no substantive source change.", source))
            continue
        if len(updated.encode("utf-8")) > MAX_PATCH_FILE_BYTES:
            issues.append(_issue("oversized_file", path, "Resulting file exceeds the PatchGate size limit."))
            continue
        before.setdefault(path, live_source)
        after[path] = updated
        modes[path] = mode
        converted.append(
            PatchEdit(
                file_path=path,
                node_target=edit.node_target or "",
                search=edit.search,
                replace=edit.replace,
                expected_content_sha256=expected,
                exact_only=True,
            )
        )

    if issues:
        _clear_gate_output(state)
        return PatchGateResult(accepted=False, issues=issues[:16])
    try:
        patch = _patch_text(before, after, modes)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        _clear_gate_output(state)
        return PatchGateResult(
            accepted=False,
            issues=[_issue("apply_failed", "", str(exc))],
        )
    if not patch.strip() or not converted:
        _clear_gate_output(state)
        return PatchGateResult(accepted=False, issues=[_issue("empty_patch", "", "Batch produces no substantive diff.")])
    manifest = [
        SnapshotManifestEntry(
            path=path,
            change="added" if before[path] is None else "modified",
            mode=modes[path],  # type: ignore[arg-type]
            content_sha256=hashlib.sha256(after[path].encode("utf-8")).hexdigest(),
            size=len(after[path].encode("utf-8")),
        )
        for path in sorted(after)
        if before[path] != after[path]
    ]
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    apply_check = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", "-"],
        input=patch.encode("utf-8"),
        check=False,
        capture_output=True,
        timeout=30,
    )
    if apply_check.returncode:
        _clear_gate_output(state)
        return PatchGateResult(
            accepted=False,
            issues=[_issue("apply_failed", "", "Generated patch failed exact git apply check.")],
        )
    gate_fingerprint = _fingerprint(
        state,
        plan,
        converted,
        manifest,
        patch_sha256,
    )
    state.patch_content = patch
    state.patch_edits = converted
    state.tool_patch_approval = ToolPatchApproval(
        base_ref=state.repo_ref,
        patch_sha256=patch_sha256,
        patch_gate_fingerprint=gate_fingerprint,
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )
    state.patch_correction_count = 0
    return PatchGateResult(accepted=True, edits=converted)


@dataclass(frozen=True)
class _ApprovedSnapshot:
    root: Path
    before: dict[str, bytes | None]
    after: dict[str, bytes]
    modes: dict[str, str]
    patch: str
    approval: ToolPatchApproval


def _apply_canonical_edit(source: str, edit: PatchEdit, plan: RepairPlan) -> str:
    if edit.node_target:
        if edit.node_target not in plan.target_symbols:
            raise ValueError("PatchGate approved node is outside the active RepairPlan")
        updated = _node_replace(source, edit.node_target, edit.replace)
        if updated is None:
            raise ValueError("PatchGate approved node target changed")
        return updated
    logical, boundaries = _normalize_with_boundaries(source)
    search, _ = _normalize_with_boundaries(edit.search)
    if not search or logical.count(search) != 1:
        raise ValueError("PatchGate approved exact search changed")
    start = logical.index(search)
    end = start + len(search)
    raw_start, raw_end = boundaries[start], boundaries[end]
    replacement = _raw_replacement(
        edit.replace,
        _newline_style(source, raw_start, raw_end),
    )
    return source[:raw_start] + replacement + source[raw_end:]


def _rebuild_approved_snapshot(state: AgentState) -> _ApprovedSnapshot:
    """Recompute every approval field from the exact live base and active plan."""
    root = _exact_root(state)
    plan = state.active_repair_plan
    if plan is None:
        raise ValueError("PatchGate approval requires an active RepairPlan")
    if not state.patch_edits:
        raise ValueError("PatchGate approval requires ordered exact edits")
    before_text: dict[str, str | None] = {}
    after_text: dict[str, str] = {}
    modes: dict[str, str] = {}
    seen_anchors: set[tuple[str, str, str]] = set()
    empty_digest = hashlib.sha256(b"").hexdigest()
    for edit in state.patch_edits:
        path = canonical_repo_path(edit.file_path)
        if path not in plan.target_files:
            raise ValueError("PatchGate approved edit is outside the active RepairPlan")
        anchor = (path, edit.node_target, edit.search)
        if anchor in seen_anchors:
            raise ValueError("PatchGate approved edit order contains a duplicate anchor")
        seen_anchors.add(anchor)
        if not edit.exact_only or not edit.expected_content_sha256:
            raise ValueError("PatchGate target preimage binding changed")
        tracked = _tree_entry(root, state.repo_ref, path)
        if tracked is None:
            _safe_leaf(root, path, must_exist=False)
            if (
                edit.search
                or edit.node_target
                or path in before_text
                or edit.expected_content_sha256 != empty_digest
            ):
                raise ValueError("PatchGate approved new-file binding changed")
            before_text[path] = None
            after_text[path] = edit.replace
            modes[path] = "100644"
            continue
        mode, base = tracked
        _safe_leaf(root, path, must_exist=True)
        live = _read_regular_no_follow(root, path)
        if live != base or hashlib.sha256(live).hexdigest() != edit.expected_content_sha256:
            raise ValueError("PatchGate target changed from its exact base preimage")
        raw_source = _decode_text(live)
        source = after_text.get(path, raw_source)
        before_text.setdefault(path, raw_source)
        after_text[path] = _apply_canonical_edit(source, edit, plan)
        modes[path] = mode

    patch = _patch_text(before_text, after_text, modes)
    if not patch.strip():
        raise ValueError("PatchGate approved edits now produce an empty patch")
    before = {
        path: None if source is None else source.encode("utf-8")
        for path, source in before_text.items()
    }
    after = {path: source.encode("utf-8") for path, source in after_text.items()}
    manifest = [
        SnapshotManifestEntry(
            path=path,
            change="added" if before[path] is None else "modified",
            mode=modes[path],  # type: ignore[arg-type]
            content_sha256=hashlib.sha256(after[path]).hexdigest(),
            size=len(after[path]),
        )
        for path in sorted(after)
        if before[path] != after[path]
    ]
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    fingerprint = _fingerprint(state, plan, state.patch_edits, manifest, patch_sha256)
    approval = ToolPatchApproval(
        base_ref=state.repo_ref,
        patch_sha256=patch_sha256,
        patch_gate_fingerprint=fingerprint,
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )
    apply_check = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", "-"],
        input=patch.encode("utf-8"),
        check=False,
        capture_output=True,
        timeout=30,
    )
    if apply_check.returncode:
        raise ValueError("PatchGate regenerated patch failed exact git apply check")
    return _ApprovedSnapshot(root, before, after, modes, patch, approval)


def _validated_snapshot(state: AgentState) -> _ApprovedSnapshot:
    current = state.tool_patch_approval
    if current is None:
        raise ValueError("PatchGate approval is missing")
    rebuilt = _rebuild_approved_snapshot(state)
    if state.patch_content != rebuilt.patch or current != rebuilt.approval:
        raise ValueError("PatchGate approval fingerprint or canonical fields changed")
    return rebuilt


def revalidate_approved_patch(state: AgentState) -> None:
    """Recheck every canonical approval binding immediately before mutation."""
    _validated_snapshot(state)


@dataclass
class _TransactionTarget:
    path: str
    parent_fd: int
    leaf: str
    before: bytes | None
    after: bytes
    mode: int
    target_fd: int | None = None
    stage: str = ""
    backup: str = ""
    written: bool = False


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 65_536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _same_identity(expected: os.stat_result, current: os.stat_result) -> bool:
    return (
        expected.st_dev == current.st_dev
        and expected.st_ino == current.st_ino
        and stat.S_ISREG(current.st_mode)
    )


def _open_parent_chain(
    root_fd: int,
    path: str,
    all_fds: list[int],
    created: list[tuple[int, str]],
) -> tuple[int, str]:
    parts = path.split("/")
    current = os.dup(root_fd)
    all_fds.append(current)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    for component in parts[:-1]:
        try:
            child = os.open(component, flags, dir_fd=current)
        except FileNotFoundError:
            os.mkdir(component, mode=0o755, dir_fd=current)
            created.append((current, component))
            child = os.open(component, flags, dir_fd=current)
        all_fds.append(child)
        current = child
    return current, parts[-1]


def _unlink_if_present(name: str, parent_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def apply_approved_patch(state: AgentState) -> list[str]:
    """Transactionally install an exact PatchGate snapshot with no-follow I/O."""
    snapshot = _validated_snapshot(state)
    root_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(snapshot.root, root_flags)
    all_fds = [root_fd]
    created_dirs: list[tuple[int, str]] = []
    targets: list[_TransactionTarget] = []
    try:
        for path in sorted(snapshot.after):
            parent_fd, leaf = _open_parent_chain(
                root_fd, path, all_fds, created_dirs
            )
            before = snapshot.before[path]
            target = _TransactionTarget(
                path=path,
                parent_fd=parent_fd,
                leaf=leaf,
                before=before,
                after=snapshot.after[path],
                mode=0o755 if snapshot.modes[path] == "100755" else 0o644,
            )
            if before is None:
                try:
                    os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("PatchGate new target identity changed")
            else:
                target.target_fd = os.open(
                    leaf,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                all_fds.append(target.target_fd)
                info = os.fstat(target.target_fd)
                if not stat.S_ISREG(info.st_mode) or _read_fd(target.target_fd) != before:
                    raise ValueError("PatchGate target preimage changed before staging")
            target.stage = f".repopilot-stage-{uuid.uuid4().hex}"
            stage_fd = os.open(
                target.stage,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                target.mode,
                dir_fd=parent_fd,
            )
            try:
                view = memoryview(target.after)
                while view:
                    written = os.write(stage_fd, view)
                    view = view[written:]
                os.fchmod(stage_fd, target.mode)
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            targets.append(target)

        # Freeze recoverable backups and validate identities before any target write.
        for target in targets:
            if target.before is None:
                try:
                    os.stat(target.leaf, dir_fd=target.parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise ValueError("PatchGate new target appeared before final write")
            assert target.target_fd is not None
            expected = os.fstat(target.target_fd)
            current = os.stat(
                target.leaf, dir_fd=target.parent_fd, follow_symlinks=False
            )
            if not _same_identity(expected, current) or _read_fd(target.target_fd) != target.before:
                raise ValueError("PatchGate target identity or preimage changed")
            target.backup = f".repopilot-backup-{uuid.uuid4().hex}"
            os.link(
                target.leaf,
                target.backup,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
            backup_info = os.stat(
                target.backup, dir_fd=target.parent_fd, follow_symlinks=False
            )
            current = os.stat(
                target.leaf, dir_fd=target.parent_fd, follow_symlinks=False
            )
            if not _same_identity(expected, backup_info) or not _same_identity(
                expected, current
            ):
                raise ValueError("PatchGate target identity changed while backing up")

        try:
            for target in targets:
                if target.before is None:
                    try:
                        os.stat(
                            target.leaf,
                            dir_fd=target.parent_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        raise ValueError(
                            "PatchGate new target appeared at final write"
                        )
                    os.link(
                        target.stage,
                        target.leaf,
                        src_dir_fd=target.parent_fd,
                        dst_dir_fd=target.parent_fd,
                        follow_symlinks=False,
                    )
                    _unlink_if_present(target.stage, target.parent_fd)
                else:
                    assert target.target_fd is not None
                    expected = os.fstat(target.target_fd)
                    current = os.stat(
                        target.leaf,
                        dir_fd=target.parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not _same_identity(expected, current)
                        or _read_fd(target.target_fd) != target.before
                    ):
                        raise ValueError(
                            "PatchGate target identity or preimage changed at final write"
                        )
                    os.replace(
                        target.stage,
                        target.leaf,
                        src_dir_fd=target.parent_fd,
                        dst_dir_fd=target.parent_fd,
                    )
                target.written = True
            for target in targets:
                live_fd = os.open(
                    target.leaf,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=target.parent_fd,
                )
                try:
                    if _read_fd(live_fd) != target.after:
                        raise OSError("PatchGate final write verification failed")
                finally:
                    os.close(live_fd)
        except Exception as original:
            rollback_errors: list[str] = []
            for target in reversed(targets):
                if not target.written:
                    continue
                try:
                    if target.before is None:
                        _unlink_if_present(target.leaf, target.parent_fd)
                    else:
                        os.replace(
                            target.backup,
                            target.leaf,
                            src_dir_fd=target.parent_fd,
                            dst_dir_fd=target.parent_fd,
                        )
                        target.backup = ""
                except OSError as exc:
                    rollback_errors.append(type(exc).__name__)
            for target in targets:
                try:
                    if target.before is None:
                        os.stat(
                            target.leaf,
                            dir_fd=target.parent_fd,
                            follow_symlinks=False,
                        )
                        rollback_errors.append("new_target_remained")
                    else:
                        fd = os.open(
                            target.leaf,
                            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=target.parent_fd,
                        )
                        try:
                            if _read_fd(fd) != target.before:
                                rollback_errors.append("preimage_mismatch")
                        finally:
                            os.close(fd)
                except FileNotFoundError:
                    if target.before is not None:
                        rollback_errors.append("target_missing")
                except OSError as exc:
                    rollback_errors.append(type(exc).__name__)
            if rollback_errors:
                raise RuntimeError(
                    "PatchGate transactional rollback could not be verified: "
                    + ",".join(rollback_errors)
                ) from original
            raise

        for target in targets:
            if target.backup:
                _unlink_if_present(target.backup, target.parent_fd)
        return [target.path for target in targets]
    finally:
        for target in targets:
            if target.stage:
                _unlink_if_present(target.stage, target.parent_fd)
            if target.backup:
                _unlink_if_present(target.backup, target.parent_fd)
        for parent_fd, name in reversed(created_dirs):
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
        for fd in reversed(all_fds):
            try:
                os.close(fd)
            except OSError:
                pass


__all__ = [
    "PatchGateIssue",
    "PatchGateResult",
    "apply_approved_patch",
    "revalidate_approved_patch",
    "validate_patch_batch",
]
