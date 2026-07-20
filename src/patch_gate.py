"""Fail-closed, non-mutating validation for model-authored verified edits."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import stat
import subprocess
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
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _node_replace(content: str, target: str, replacement: str) -> str | None:
    span = locate_node_span(content, target)
    if span is None:
        return None
    start, end, indent = span
    first = next((line for line in replacement.split("\n") if line.strip()), "")
    body = reindent(replacement, indent - leading_spaces(first))
    if not body.endswith("\n"):
        body += "\n"
    return content[:start] + body + content[end:]


def _node_target_count(content: str, target: str) -> int:
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


def _patch_text(before: dict[str, str | None], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for path in sorted(after):
        old = before[path]
        if old == after[path]:
            continue
        chunks.append(f"diff --git a/{path} b/{path}\n")
        if old is None:
            chunks.append("new file mode 100644\n")
        chunks.extend(
            difflib.unified_diff(
                [] if old is None else old.splitlines(keepends=True),
                after[path].splitlines(keepends=True),
                fromfile="/dev/null" if old is None else f"a/{path}",
                tofile=f"b/{path}",
                lineterm="\n",
            )
        )
    return "".join(chunks)


def _fingerprint(state: AgentState, plan: RepairPlan, batch: VerifiedEditBatch) -> str:
    payload = {
        "base_ref": state.repo_ref.lower(),
        "plan": plan.model_dump(mode="json"),
        "batch": batch.model_dump(mode="json"),
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
        return PatchGateResult(accepted=False, issues=[_issue("apply_failed", "", str(exc))])
    if state.active_repair_plan is None or state.active_repair_plan != plan:
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
            count = source.count(edit.search)
            if count == 0:
                window = closest_region(source, edit.search, max_chars=MAX_CORRECTION_CONTEXT_CHARS)
                issues.append(_issue("search_missing", path, "Exact search block is missing; fuzzy matching is disabled.", window))
                continue
            if count != 1:
                window = closest_region(source, edit.search, max_chars=MAX_CORRECTION_CONTEXT_CHARS)
                issues.append(_issue("target_ambiguous", path, "Exact search block is not unique.", window))
                continue
            updated = source.replace(edit.search, edit.replace, 1)
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
        state.tool_patch_approval = None
        return PatchGateResult(accepted=False, issues=issues[:16])
    patch = _patch_text(before, after)
    if not patch.strip() or not converted:
        state.tool_patch_approval = None
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
    gate_fingerprint = _fingerprint(state, plan, batch)
    state.patch_content = patch
    state.patch_edits = converted
    state.tool_patch_approval = ToolPatchApproval(
        base_ref=state.repo_ref,
        patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        patch_gate_fingerprint=gate_fingerprint,
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )
    state.patch_correction_count = 0
    return PatchGateResult(accepted=True, edits=converted)


def revalidate_approved_patch(state: AgentState) -> None:
    """Recheck the accepted preimages and approval immediately before mutation."""
    root = _exact_root(state)
    approval = state.tool_patch_approval
    if approval is None or approval.base_ref.lower() != state.repo_ref.lower():
        raise ValueError("PatchGate approval is missing or has the wrong base")
    if hashlib.sha256(state.patch_content.encode("utf-8")).hexdigest() != approval.patch_sha256:
        raise ValueError("PatchGate approval patch fingerprint changed")
    if tool_manifest_fingerprint(approval.changed_manifest) != approval.manifest_fingerprint:
        raise ValueError("PatchGate approval manifest changed")
    before: dict[str, str | None] = {}
    after: dict[str, str] = {}
    modes: dict[str, str] = {}
    seen_anchors: set[tuple[str, str, str]] = set()
    for edit in state.patch_edits:
        path = canonical_repo_path(edit.file_path)
        anchor = (path, edit.node_target, edit.search)
        if anchor in seen_anchors or (
            state.active_repair_plan is not None
            and path not in state.active_repair_plan.target_files
        ):
            raise ValueError("PatchGate approved edit scope changed")
        seen_anchors.add(anchor)
        tracked = _tree_entry(root, state.repo_ref, path)
        if tracked is None:
            _safe_leaf(root, path, must_exist=False)
            digest = hashlib.sha256(b"").hexdigest()
            if edit.search or edit.node_target or path in before:
                raise ValueError("PatchGate approved new-file anchor changed")
            before[path] = None
            after[path] = edit.replace
            modes[path] = "100644"
        else:
            _safe_leaf(root, path, must_exist=True)
            live = _read_regular_no_follow(root, path)
            if live != tracked[1]:
                raise ValueError("PatchGate target changed from its exact base preimage")
            digest = hashlib.sha256(live).hexdigest()
            live_source = _decode_text(live)
            source = after.get(path, live_source)
            before.setdefault(path, live_source)
            modes[path] = tracked[0]
            if edit.node_target:
                updated = _node_replace(source, edit.node_target, edit.replace)
                if updated is None:
                    raise ValueError("PatchGate approved node target changed")
            elif edit.search and source.count(edit.search) == 1:
                updated = source.replace(edit.search, edit.replace, 1)
            else:
                raise ValueError("PatchGate approved exact search changed")
            after[path] = updated
        if not edit.exact_only or digest != edit.expected_content_sha256:
            raise ValueError("PatchGate target preimage binding changed")
    regenerated_patch = _patch_text(before, after)
    if hashlib.sha256(regenerated_patch.encode("utf-8")).hexdigest() != approval.patch_sha256:
        raise ValueError("PatchGate approved edits no longer match the approved patch")
    regenerated_manifest = [
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
    if regenerated_manifest != approval.changed_manifest:
        raise ValueError("PatchGate approved edits no longer match the approved manifest")


__all__ = [
    "PatchGateIssue",
    "PatchGateResult",
    "revalidate_approved_patch",
    "validate_patch_batch",
]
