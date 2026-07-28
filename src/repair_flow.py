"""Two-stage, exact-context repair flow for the escalation model."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .async_safety import CancellationDrainError
from .escalation import (
    EscalationPacket,
    build_escalation_packet,
    prepare_repair_plan_packet,
    record_model_invocation,
    render_escalation_packet,
)
from .evidence import EvidenceStore
from .llm import llm_call
from .local_search import is_sensitive_repo_path
from .model_policy import apply_escalation, should_escalate
from .model_provider import redact_secrets
from .outcome_summary import MAX_OUTCOME_SUMMARY_CHARS, OUTCOME_SUMMARY_SECTION
from .patch_match import locate_node_span
from .reasoning_loop import (
    ReasoningStop,
    route_reasoning_tool,
    validate_reasoning_response,
)
from .repo_paths import canonical_repo_path
from .state import (
    AgentState,
    Evidence,
    PatchEdit,
    RepairPlan,
    VerifiedEditBatch,
    _estimate_tokens,
)
from .summary_safety import sanitize_summary_text
from .tool_router import ToolRouteResult, route_tool_intent

TARGET_CONTEXT_CONTENT_LIMIT = 8_000
TARGET_CONTEXT_TOTAL_LIMIT = 24_000
TARGET_CONTEXT_FILE_LIMIT = 8
TARGET_CONTEXT_SYMBOL_LIMIT = 16
REPAIR_PROMPT_LIMIT = 120_000
REPAIR_CONTEXT_CORRECTION_LIMIT = 500
SURROUNDING_LINES = 3

_REPAIR_CONTEXT_CORRECTION_SECTION = "REPAIR TARGET CORRECTION:"

_TEXT_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".rst",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_EVALUATOR_FIELD_RE = re.compile(
    r"(?i)\b(?:gold[_ -]?patch|test[_ -]?patch|FAIL_TO_PASS|PASS_TO_PASS)\b"
)
_UNIFIED_DIFF_RE = re.compile(
    r"(?im)^\s*(?:diff --git\b|@@\s|---\s+(?:a/|/dev/null)|"
    r"\+\+\+\s+(?:b/|/dev/null)|\*\*\*\s+(?:begin|end)\s+patch\b)"
)

REPAIR_PLAN_SYSTEM = (
    "Return ONLY one discriminated JSON variant: kind='tool' with one tool_intent, "
    "kind='repair_plan' containing exactly these keys: root_cause, "
    "target_files, target_symbols, required_behavior, regression_test_strategy, "
    "rejected_approaches, or kind='stop'. Identify a bounded implementation plan from the supplied "
    "EscalationPacket. target_files must be repository-relative paths and "
    "target_symbols must use exact dotted names when applicable."
)

VERIFIED_EDIT_SYSTEM = (
    "Return ONLY one discriminated JSON variant: kind='tool' with one tool_intent, "
    "kind='verified_edits' with key edits, or kind='stop'. Each edit must contain file_path, "
    "node_target (string or null), search, replace, and intent. Use only a target "
    "file and evidence shown in the user payload. Prefer one unique node_target; "
    "otherwise copy a search string verbatim from that file's target evidence. "
    "Do not return a unified diff or infer unseen file content."
)

VERIFIED_EDIT_CORRECTION_SYSTEM = (
    "Return ONLY one discriminated JSON variant: kind='tool' with one tool_intent, "
    "kind='verified_edits' with key edits, or kind='stop'. Correct the previous verified "
    "edit batch using the exact PatchGate issue code and real bounded code window. "
    "Use only RepairPlan target files. Use one unique node_target or copy a search "
    "block verbatim. Never use fuzzy matching or a unified diff."
)


class RepairContextError(ValueError):
    """The requested context or model-authored anchor failed closed."""


@dataclass(frozen=True)
class _TargetSnapshot:
    file_path: str
    content: str | None
    content_sha256: str
    is_new: bool


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _validate_checkout(state: AgentState) -> Path:
    if not state.repo_path or not state.repo_ref:
        raise RepairContextError("exact checkout path and base commit are required")
    root = Path(state.repo_path).resolve()
    if not root.is_dir():
        raise RepairContextError("exact checkout does not exist")
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != state.repo_ref:
        raise RepairContextError("checkout HEAD does not match the exact base commit")
    return root


def _decode_utf8(data: bytes) -> str:
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairContextError("target file must be readable UTF-8 text") from exc
    # Match EvidenceStore and Path.read_text() universal-newline semantics while
    # retaining the raw-byte digest as the immutable checkout preimage.
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


def _read_regular_no_follow(root: Path, relative: str) -> bytes:
    """Read a regular file without following any checkout-relative symlink."""
    parts = relative.split("/")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(
                component,
                directory_flags | no_follow,
                dir_fd=current,
            )
            descriptors.append(current)
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | no_follow,
            dir_fd=current,
        )
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RepairContextError("target file is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RepairContextError("target file changed while reading its preimage")
        return b"".join(chunks)
    except RepairContextError:
        raise
    except OSError as exc:
        raise RepairContextError(
            "target file is missing, symlinked, or not a regular checkout file"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _assert_intentional_new_target(root: Path, relative: str) -> None:
    """Prove every existing parent is a real directory and the leaf is absent."""
    current = root
    parts = relative.split("/")
    for index, component in enumerate(parts):
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(status.st_mode):
            raise RepairContextError("target file symlink escapes the exact checkout")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(status.st_mode):
                raise RepairContextError("target file parent is not a directory")
        else:
            raise RepairContextError("target file is not tracked at the exact checkout")


def _target_snapshot(root: Path, relative: str) -> _TargetSnapshot:
    path = canonical_repo_path(relative)
    if is_sensitive_repo_path(path):
        raise RepairContextError("target file is a sensitive repository path")
    tracked = _git(root, "ls-files", "--error-unmatch", "--", path)
    if tracked.returncode == 0:
        data = _read_regular_no_follow(root, path)
        return _TargetSnapshot(
            file_path=path,
            content=_decode_utf8(data),
            content_sha256=hashlib.sha256(data).hexdigest(),
            is_new=False,
        )
    _assert_intentional_new_target(root, path)
    if Path(path).suffix.lower() not in _TEXT_SOURCE_SUFFIXES:
        raise RepairContextError(
            "target file is missing or not an intentional new text file"
        )
    return _TargetSnapshot(
        file_path=path,
        content=None,
        content_sha256=hashlib.sha256(b"").hexdigest(),
        is_new=True,
    )


def _python_symbol_spans(source: str, requested: str) -> list[tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RepairContextError("Python target file does not parse") from exc
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    spans: list[tuple[int, int]] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [*stack, child.name]
                qualname = ".".join(names)
                if qualname == requested:
                    start_line = child.lineno
                    decorators = getattr(child, "decorator_list", ())
                    if decorators:
                        start_line = min(
                            start_line, *(item.lineno for item in decorators)
                        )
                    end_line = child.end_lineno or child.lineno
                    spans.append((offsets[start_line - 1], offsets[end_line]))
                walk(child, names)
            else:
                walk(child, stack)

    walk(tree, [])
    return spans


def _definition_window(source: str, span: tuple[int, int]) -> str:
    start, end = span
    definition = source[start:end]
    if len(definition) > TARGET_CONTEXT_CONTENT_LIMIT:
        raise RepairContextError("complete Python definition exceeds context budget")
    line_starts = [0]
    for match in re.finditer("\n", source):
        line_starts.append(match.end())
    start_line = max(0, source[:start].count("\n") - SURROUNDING_LINES)
    end_line = min(len(line_starts), source[:end].count("\n") + SURROUNDING_LINES + 1)
    window_start = line_starts[start_line]
    window_end = line_starts[end_line] if end_line < len(line_starts) else len(source)
    window = source[window_start:window_end]
    while len(window) > TARGET_CONTEXT_CONTENT_LIMIT and window_start < start:
        next_newline = source.find("\n", window_start)
        window_start = start if next_newline < 0 else min(start, next_newline + 1)
        window = source[window_start:window_end]
    while len(window) > TARGET_CONTEXT_CONTENT_LIMIT and window_end > end:
        previous_newline = source.rfind("\n", end, max(end, window_end - 1))
        window_end = end if previous_newline < 0 else max(end, previous_newline + 1)
        window = source[window_start:window_end]
    if len(window) > TARGET_CONTEXT_CONTENT_LIMIT:
        return definition
    return window


def _fallback_window(source: str, symbol: str | None = None) -> str:
    if len(source) <= TARGET_CONTEXT_CONTENT_LIMIT:
        return source
    if symbol:
        needle = symbol.rsplit(".", 1)[-1]
        offset = source.find(needle)
        if offset >= 0:
            half = TARGET_CONTEXT_CONTENT_LIMIT // 2
            start = max(0, offset - half)
            end = min(len(source), start + TARGET_CONTEXT_CONTENT_LIMIT)
            start = max(0, end - TARGET_CONTEXT_CONTENT_LIMIT)
            return source[start:end]
    return source[:TARGET_CONTEXT_CONTENT_LIMIT]


def _build_target_context_with_snapshots(
    state: AgentState,
    plan: RepairPlan,
) -> tuple[list[Evidence], dict[str, _TargetSnapshot]]:
    """Resolve targets from fresh no-follow reads and return exact preimages."""
    if len(plan.target_files) > TARGET_CONTEXT_FILE_LIMIT:
        raise RepairContextError("too many target files for context budget")
    if len(plan.target_symbols) > TARGET_CONTEXT_SYMBOL_LIMIT:
        raise RepairContextError("too many target symbols for context budget")
    root = _validate_checkout(state)
    targets: dict[str, _TargetSnapshot] = {}
    for relative in plan.target_files:
        targets[relative] = _target_snapshot(root, relative)

    pending: list[dict[str, str | None]] = []
    represented_files: set[str] = set()
    for symbol in plan.target_symbols:
        matches: list[tuple[str, str, tuple[int, int] | None]] = []
        for relative, snapshot in targets.items():
            source = snapshot.content
            if source is None:
                continue
            if Path(relative).suffix.lower() == ".py":
                matches.extend(
                    (relative, source, span)
                    for span in _python_symbol_spans(source, symbol)
                )
            elif source.count(symbol.rsplit(".", 1)[-1]) == 1:
                matches.append((relative, source, None))
        if len(matches) != 1:
            raise RepairContextError(
                f"target symbol {symbol!r} is missing or not unique in RepairPlan files"
            )
        relative, source, span = matches[0]
        content = (
            _definition_window(source, span)
            if span
            else _fallback_window(source, symbol)
        )
        pending.append(
            {
                "tool": "repair_context",
                "summary": f"Exact target context for {relative}:{symbol}",
                "content": content,
                "file_path": relative,
                "symbol": symbol,
            }
        )
        represented_files.add(relative)

    for relative, snapshot in targets.items():
        source = snapshot.content
        if relative in represented_files:
            continue
        pending.append(
            {
                "tool": "repair_context",
                "summary": (
                    f"RepairPlan intentional new text file: {relative}"
                    if source is None
                    else f"Bounded exact target file context for {relative}"
                ),
                "content": "" if source is None else _fallback_window(source),
                "file_path": relative,
                "symbol": None,
            }
        )

    if any(
        _EVALUATOR_FIELD_RE.search(str(item.get("content") or "")) for item in pending
    ):
        raise RepairContextError("target context contains an evaluator-only field")

    # Persisted Evidence is untrusted input. Building the target prompt from an
    # empty isolated store prevents a forged ID/fingerprint collision from
    # substituting attacker-controlled content for the fresh checkout read.
    temporary = state.model_copy(deep=True)
    temporary.evidence = []
    store = EvidenceStore(
        temporary,
        max_items=30,
        max_content_chars=TARGET_CONTEXT_CONTENT_LIMIT,
    )
    result: list[Evidence] = []
    for item in pending:
        added = store.add(**item)  # type: ignore[arg-type]
        if added.evidence is None:
            raise RepairContextError("evidence store capacity exhausted")
        result.append(added.evidence)
    selected = store.select(
        [item.evidence_id for item in result],
        max_total_chars=TARGET_CONTEXT_TOTAL_LIMIT,
    )
    if [item.evidence_id for item in selected] != [item.evidence_id for item in result]:
        raise RepairContextError("target evidence exceeds total context budget")
    collision_ids = {item.evidence_id for item in result}
    collision_fingerprints = {item.fingerprint for item in result}
    preserved = [
        item
        for item in state.evidence
        if item.evidence_id not in collision_ids
        and item.fingerprint not in collision_fingerprints
    ]
    available = max(0, 30 - len(result))
    state.evidence = [*preserved[:available], *result]
    return result, targets


def build_target_context(state: AgentState, plan: RepairPlan) -> list[Evidence]:
    """Resolve every plan target to bounded exact text in the prepared checkout."""
    evidence, _snapshots = _build_target_context_with_snapshots(state, plan)
    return evidence


def read_exact_checkout_text(state: AgentState, file_path: str) -> str:
    """Read one confined regular UTF-8 file from the exact prepared checkout."""
    root = _validate_checkout(state)
    path = canonical_repo_path(file_path)
    if is_sensitive_repo_path(path):
        raise RepairContextError("target file is a sensitive repository path")
    return _decode_utf8(_read_regular_no_follow(root, path))


def resolve_search_target_symbol(
    state: AgentState,
    file_path: str,
    search: str,
) -> str | None:
    """Best-effort compatibility wrapper used by legacy EXECUTE paths."""
    try:
        return resolve_search_target_symbol_strict(state, file_path, search)
    except (OSError, RepairContextError):
        return None


def resolve_search_target_symbol_strict(
    state: AgentState,
    file_path: str,
    search: str,
) -> str | None:
    """Resolve a symbol while preserving exact-checkout environment failures."""
    if not search or not file_path.endswith(".py"):
        return None
    try:
        source = read_exact_checkout_text(state, file_path)
    except RepairContextError:
        raise
    except OSError as exc:
        raise RepairContextError("exact checkout target read failed") from exc
    if source.count(search) != 1:
        return None
    line = source.count("\n", 0, source.index(search)) + 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    candidates: list[tuple[int, str]] = []

    def visit(node: ast.AST, prefix: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            child_prefix = prefix
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                end_line = getattr(child, "end_lineno", child.lineno)
                child_prefix = (*prefix, child.name)
                if child.lineno <= line <= end_line:
                    candidates.append((len(child_prefix), ".".join(child_prefix)))
            visit(child, child_prefix)

    visit(tree)
    return max(candidates)[1] if candidates else None


def _safe_generated_text(value: str) -> str:
    redacted = redact_secrets(value)
    return _EVALUATOR_FIELD_RE.sub("[REDACTED_EVALUATOR_FIELD]", redacted)


def _safe_plan(plan: RepairPlan) -> RepairPlan:
    payload = plan.model_dump()
    for key in (
        "root_cause",
        "required_behavior",
        "regression_test_strategy",
    ):
        payload[key] = _safe_generated_text(payload[key])
    payload["target_symbols"] = [
        _safe_generated_text(item) for item in payload["target_symbols"]
    ]
    payload["rejected_approaches"] = [
        _safe_generated_text(item) for item in payload["rejected_approaches"]
    ]
    if any(
        _EVALUATOR_FIELD_RE.search(item)
        for item in [*payload["target_files"], *payload["target_symbols"]]
    ):
        raise RepairContextError("RepairPlan target contains an evaluator-only field")
    return RepairPlan.model_validate(payload)


SchemaT = TypeVar("SchemaT", bound=BaseModel)
ResultT = TypeVar("ResultT")


async def _call_schema(
    state: AgentState,
    *,
    system: str,
    user: str,
    schema: type[SchemaT],
    semantic_validate: Callable[[SchemaT], ResultT],
    outcome_kind: str,
    router: Callable[..., Awaitable[ToolRouteResult]],
    policy_hook: Callable[[AgentState], None] | None = None,
    reprompt: Callable[[tuple[str, ...]], str] | None = None,
    tool_counter: list[int] | None = None,
) -> ResultT:
    counter = tool_counter if tool_counter is not None else [0]
    current_user = user
    outcome_fields = set(schema.model_fields)
    while True:
        apply_escalation(state, should_escalate(state))
        if policy_hook is not None:
            policy_hook(state)
        model = state.active_model
        provider = state.active_provider
        started = time.monotonic()
        response_text = ""
        tool_step = None
        try:
            raw = await llm_call(
                system,
                current_user,
                model=model,
                provider=provider,
            )
            if not isinstance(raw, dict):
                raise ValueError("structured response must be a JSON object")
            response_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            response_kind = validate_reasoning_response(
                raw,
                outcome_kind=outcome_kind,
                outcome_fields=outcome_fields,
            )
            if response_kind == "stop":
                raise ReasoningStop(str(raw.get("stop_reason") or "model_stop"))
            tool_step = await route_reasoning_tool(
                state,
                raw,
                node="plan_fix",
                calls_this_round=counter[0],
                router=router,
            )
            if tool_step.handled:
                result = None
            else:
                payload = {key: value for key, value in raw.items() if key != "kind"}
                parsed = schema.model_validate(payload)
                result = semantic_validate(parsed)
        except CancellationDrainError:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - started
            input_tokens = _estimate_tokens(system, current_user)
            output_tokens = _estimate_tokens(response_text) if response_text else 0
            record_model_invocation(
                state,
                model=model,
                provider=provider,
                node="plan_fix",
                elapsed_seconds=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status=(
                    "invalid_response"
                    if isinstance(
                        exc,
                        (
                            ValidationError,
                            RepairContextError,
                            ReasoningStop,
                            ValueError,
                        ),
                    )
                    else "error"
                ),
                error=exc,
            )
            state.token_usage += input_tokens + output_tokens
            raise
        elapsed = time.monotonic() - started
        input_tokens = _estimate_tokens(system, current_user)
        output_tokens = _estimate_tokens(response_text)
        record_model_invocation(
            state,
            model=model,
            provider=provider,
            node="plan_fix",
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="ok",
        )
        state.token_usage += input_tokens + output_tokens
        if tool_step is not None and tool_step.handled:
            if tool_step.stop_reason:
                raise ReasoningStop(
                    tool_step.stop_reason,
                    code=tool_step.stop_reason,
                )
            counter[0] += 1
            current_user = (
                reprompt(tool_step.evidence_ids)
                if reprompt is not None
                else current_user
            )
            continue
        return result  # type: ignore[return-value]


def _validate_batch(
    state: AgentState,
    plan: RepairPlan,
    batch: VerifiedEditBatch,
    evidence: list[Evidence],
    snapshots: dict[str, _TargetSnapshot],
) -> None:
    root = _validate_checkout(state)
    plan_files = set(plan.target_files)
    evidence_by_file: dict[str, list[Evidence]] = {}
    for item in evidence:
        if item.file_path:
            evidence_by_file.setdefault(item.file_path, []).append(item)
    anchors: set[tuple[str, str, str]] = set()
    for edit in batch.edits:
        if edit.file_path not in plan_files:
            raise RepairContextError("verified edit file is outside the RepairPlan")
        expected = snapshots.get(edit.file_path)
        if expected is None:
            raise RepairContextError("verified edit lacks an exact target preimage")
        current = _target_snapshot(root, edit.file_path)
        if (
            current.is_new != expected.is_new
            or current.content_sha256 != expected.content_sha256
        ):
            raise RepairContextError("target file changed from its exact preimage")
        if _UNIFIED_DIFF_RE.search(edit.search) or _UNIFIED_DIFF_RE.search(
            edit.replace
        ):
            raise RepairContextError(
                "unified diff content is not a verified edit anchor"
            )
        if (
            redact_secrets(edit.search) != edit.search
            or redact_secrets(edit.replace) != edit.replace
        ):
            raise RepairContextError("verified edit contains credential-shaped text")
        anchor = (edit.file_path, edit.node_target or "", edit.search)
        if anchor in anchors:
            raise RepairContextError("verified edit anchors must be unique")
        anchors.add(anchor)
        edit._expected_content_sha256 = expected.content_sha256
        edit._exact_only = True
        if current.is_new:
            if edit.node_target or edit.search:
                raise RepairContextError(
                    "intentional new text file cannot use an existing anchor"
                )
            continue
        source = current.content
        if source is None:  # defensive narrowing; existing targets always have content
            raise RepairContextError("existing target file lacks an exact preimage")
        file_evidence = evidence_by_file.get(edit.file_path, [])
        if edit.node_target:
            if edit.node_target not in plan.target_symbols:
                raise RepairContextError(
                    "verified edit node is outside RepairPlan symbols"
                )
            if locate_node_span(source, edit.node_target) is None:
                raise RepairContextError(
                    "verified edit node target is missing or ambiguous"
                )
            if not any(item.symbol == edit.node_target for item in file_evidence):
                raise RepairContextError(
                    "verified edit node lacks exact target evidence"
                )
            try:
                replacement = ast.parse(textwrap.dedent(edit.replace))
            except SyntaxError as exc:
                raise RepairContextError(
                    "node replacement is not valid Python"
                ) from exc
            if len(replacement.body) != 1 or not isinstance(
                replacement.body[0],
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                raise RepairContextError(
                    "node replacement must be one complete definition"
                )
            definition = replacement.body[0]
            if definition.name != edit.node_target.rsplit(".", 1)[-1]:
                raise RepairContextError(
                    "node replacement does not preserve its target symbol"
                )
        else:
            if not edit.search:
                raise RepairContextError(
                    "existing-file verified edit requires an exact anchor"
                )
            if source.count(edit.search) != 1:
                raise RepairContextError(
                    "verified edit search anchor is missing or not unique"
                )
            if not any(edit.search in item.content for item in file_evidence):
                raise RepairContextError(
                    "verified edit search anchor was not in target evidence"
                )


def verified_edits_to_patch_edits(
    batch: VerifiedEditBatch,
    *,
    state: AgentState,
) -> list[PatchEdit]:
    """Convert already-validated existing-file edits to the legacy executor type."""
    root = _validate_checkout(state)
    edits: list[PatchEdit] = []
    for edit in batch.edits:
        if not edit._exact_only or not edit._expected_content_sha256:
            raise RepairContextError(
                "verified edit is missing its exact preimage binding"
            )
        current = _target_snapshot(root, edit.file_path)
        if current.content_sha256 != edit._expected_content_sha256:
            raise RepairContextError("target file changed from its exact preimage")
        if not edit.search and not edit.node_target:
            raise RepairContextError(
                "intentional new text file edits require PatchGate before conversion"
            )
        edits.append(
            PatchEdit(
                file_path=edit.file_path,
                node_target=edit.node_target or "",
                search=edit.search,
                replace=edit.replace,
                expected_content_sha256=edit._expected_content_sha256,
                exact_only=True,
            )
        )
    return edits


def _render_default_repair_plan_reprompt(
    state: AgentState,
    initial_packet: EscalationPacket,
    evidence_ids: tuple[str, ...],
    suffix: str = "",
) -> str:
    """Render fresh tool evidence followed by deterministic source evidence."""
    rendered = render_escalation_packet(
        prepare_repair_plan_packet(
            state,
            initial_packet,
            evidence_ids=evidence_ids,
        )
    )
    return f"{rendered}{suffix}"


def _first_stage_prompt_suffix(prompt: str) -> str:
    """Discard a caller-supplied packet and retain only its bounded suffix."""
    try:
        payload, offset = json.JSONDecoder().raw_decode(prompt)
        EscalationPacket.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise RepairContextError(
            "custom repair prompt must start with one valid escalation packet"
        ) from exc
    return _validated_first_stage_suffix(prompt[offset:])


def _validated_first_stage_suffix(suffix: str) -> str:
    """Allow only bounded rolling-summary and repair-correction sections."""
    if not suffix:
        return ""
    summary_prefix = f"\n\n{OUTCOME_SUMMARY_SECTION}\n"
    correction_prefix = f"\n\n{_REPAIR_CONTEXT_CORRECTION_SECTION}\n"
    summary = ""
    correction = ""
    if suffix.startswith(summary_prefix):
        remainder = suffix[len(summary_prefix) :]
        if correction_prefix in remainder:
            summary, correction_body = remainder.split(correction_prefix, 1)
            correction = f"{correction_prefix}{correction_body}"
        else:
            summary = remainder
    elif suffix.startswith(correction_prefix):
        correction = suffix
    else:
        raise RepairContextError("custom repair prompt suffix is not allowlisted")

    if summary:
        if (
            len(summary) > MAX_OUTCOME_SUMMARY_CHARS
            or sanitize_summary_text(summary, len(summary)) != summary
            or any(marker in summary for marker in ("{", "}"))
            or OUTCOME_SUMMARY_SECTION in summary
            or _REPAIR_CONTEXT_CORRECTION_SECTION in summary
        ):
            raise RepairContextError("custom repair summary suffix is invalid")
    elif suffix.startswith(summary_prefix):
        raise RepairContextError("custom repair summary suffix is empty")

    if correction:
        correction_body = correction[len(correction_prefix) :]
        if (
            not correction_body
            or len(correction) > REPAIR_CONTEXT_CORRECTION_LIMIT
            or sanitize_summary_text(correction_body, len(correction_body))
            != correction_body
            or any(marker in correction_body for marker in ("{", "}"))
            or OUTCOME_SUMMARY_SECTION in correction_body
            or _REPAIR_CONTEXT_CORRECTION_SECTION in correction_body
        ):
            raise RepairContextError("custom repair correction suffix is invalid")
    return suffix


async def generate_opus_repair(
    state: AgentState,
    packet: EscalationPacket,
    *,
    first_stage_prompt: str | None = None,
    first_stage_suffix: str = "",
    validate_edits: bool = True,
    router: Callable[..., Awaitable[ToolRouteResult]] = route_tool_intent,
    policy_hook: Callable[[AgentState], None] | None = None,
    tool_counter: list[int] | None = None,
    first_stage_reprompt: Callable[[tuple[str, ...]], str] | None = None,
) -> tuple[RepairPlan, VerifiedEditBatch]:
    """Generate patch-free intent, resolve exact context, then request edits."""
    if state.active_provider != "escalation" or not state.escalated:
        raise RepairContextError("two-stage repair requires one-way escalation")
    if packet.base_commit != state.repo_ref:
        raise RepairContextError("EscalationPacket base commit does not match checkout")
    packet = prepare_repair_plan_packet(state, packet)
    rendered_packet = render_escalation_packet(packet)
    if first_stage_prompt is not None and first_stage_suffix:
        raise RepairContextError(
            "repair plan prompt cannot combine a full override with a suffix"
        )
    if first_stage_prompt is not None:
        first_stage_suffix = _first_stage_prompt_suffix(first_stage_prompt)
    else:
        first_stage_suffix = _validated_first_stage_suffix(first_stage_suffix)
    plan_user = f"{rendered_packet}{first_stage_suffix}"
    if len(plan_user) > REPAIR_PROMPT_LIMIT:
        raise RepairContextError("repair plan prompt exceeds strict context budget")
    reasoning_tool_counter = tool_counter if tool_counter is not None else [0]
    state._reasoning_tool_counter = reasoning_tool_counter

    def validate_plan(
        candidate: RepairPlan,
    ) -> tuple[
        RepairPlan,
        list[Evidence],
        dict[str, _TargetSnapshot],
    ]:
        safe_plan = _safe_plan(candidate)
        evidence, snapshots = _build_target_context_with_snapshots(state, safe_plan)
        return safe_plan, evidence, snapshots

    plan, evidence, snapshots = await _call_schema(
        state,
        system=REPAIR_PLAN_SYSTEM,
        user=plan_user,
        schema=RepairPlan,
        semantic_validate=validate_plan,
        outcome_kind="repair_plan",
        router=router,
        policy_hook=policy_hook,
        reprompt=lambda evidence_ids: _render_default_repair_plan_reprompt(
            state,
            packet,
            evidence_ids,
            (
                _first_stage_prompt_suffix(first_stage_reprompt(evidence_ids))
                if first_stage_reprompt is not None
                else first_stage_suffix
            ),
        ),
        tool_counter=reasoning_tool_counter,
    )
    payload = {
        "escalation_packet": json.loads(rendered_packet),
        "repair_plan": plan.model_dump(mode="json"),
        "target_evidence": [
            {
                "evidence_id": item.evidence_id,
                "file_path": item.file_path,
                "symbol": item.symbol,
                "content": item.content,
            }
            for item in evidence
        ],
    }
    user = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    if len(user) > REPAIR_PROMPT_LIMIT:
        raise RepairContextError("verified edit prompt exceeds strict context budget")
    should_validate_edits = validate_edits

    def validate_edit_batch(candidate: VerifiedEditBatch) -> VerifiedEditBatch:
        if should_validate_edits:
            _validate_batch(state, plan, candidate, evidence, snapshots)
        return candidate

    batch = await _call_schema(
        state,
        system=VERIFIED_EDIT_SYSTEM,
        user=user,
        schema=VerifiedEditBatch,
        semantic_validate=validate_edit_batch,
        outcome_kind="verified_edits",
        router=router,
        policy_hook=policy_hook,
        reprompt=lambda evidence_ids: json.dumps(
            {
                **payload,
                "escalation_packet": json.loads(
                    render_escalation_packet(
                        build_escalation_packet(state, evidence_ids=evidence_ids)
                    )
                ),
                "new_evidence": [
                    item.model_dump(mode="json")
                    for item in EvidenceStore(state).select(list(evidence_ids))
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        tool_counter=reasoning_tool_counter,
    )
    return plan, batch


async def request_verified_edit_correction(
    state: AgentState,
    plan: RepairPlan,
    previous_batch: VerifiedEditBatch,
    issues: list[object],
    *,
    router: Callable[..., Awaitable[ToolRouteResult]] = route_tool_intent,
    policy_hook: Callable[[AgentState], None] | None = None,
    tool_counter: list[int] | None = None,
) -> VerifiedEditBatch:
    """Ask the same active model for one bounded, local PatchGate correction."""
    if state.active_provider != "escalation" or not state.escalated:
        raise RepairContextError("verified edit correction requires active escalation")
    safe_issues: list[dict[str, str]] = []
    for issue in issues[:16]:
        safe_issues.append(
            {
                "code": str(getattr(issue, "code", "apply_failed"))[:64],
                "file_path": (
                    canonical_repo_path(str(getattr(issue, "file_path", "")))
                    if str(getattr(issue, "file_path", ""))
                    else ""
                ),
                "message": _safe_generated_text(str(getattr(issue, "message", "")))[
                    :500
                ],
                "real_code_window": _safe_generated_text(
                    str(getattr(issue, "correction_context", ""))
                )[:1_200],
            }
        )
    payload = {
        "repair_plan": plan.model_dump(mode="json"),
        "patch_gate_issues": safe_issues,
        "previous_edits": [
            {
                "file_path": edit.file_path,
                "node_target": edit.node_target,
                "search": edit.search[:8_000],
                "replace": edit.replace[:8_000],
                "intent": edit.intent,
            }
            for edit in previous_batch.edits
        ],
    }
    user = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    if len(user) > 40_000:
        raise RepairContextError(
            "verified edit correction exceeds strict context budget"
        )
    return await _call_schema(
        state,
        system=VERIFIED_EDIT_CORRECTION_SYSTEM,
        user=user,
        schema=VerifiedEditBatch,
        semantic_validate=lambda candidate: candidate,
        outcome_kind="verified_edits",
        router=router,
        policy_hook=policy_hook,
        reprompt=lambda evidence_ids: json.dumps(
            {
                **payload,
                "new_evidence": [
                    item.model_dump(mode="json")
                    for item in EvidenceStore(state).select(list(evidence_ids))
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        tool_counter=(
            tool_counter if tool_counter is not None else state._reasoning_tool_counter
        ),
    )
