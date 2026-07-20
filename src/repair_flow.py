"""Two-stage, exact-context repair flow for the escalation model."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .escalation import (
    EscalationPacket,
    record_model_invocation,
    render_escalation_packet,
)
from .evidence import EvidenceStore
from .llm import llm_call
from .local_search import is_sensitive_repo_path
from .model_provider import redact_secrets
from .patch_match import locate_node_span
from .repo_paths import canonical_repo_path
from .state import (
    AgentState,
    Evidence,
    PatchEdit,
    RepairPlan,
    VerifiedEditBatch,
    _estimate_tokens,
)

TARGET_CONTEXT_CONTENT_LIMIT = 8_000
TARGET_CONTEXT_TOTAL_LIMIT = 24_000
TARGET_CONTEXT_FILE_LIMIT = 8
TARGET_CONTEXT_SYMBOL_LIMIT = 16
REPAIR_PROMPT_LIMIT = 120_000
SURROUNDING_LINES = 3

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
_UNIFIED_DIFF_RE = re.compile(r"(?m)^(?:diff --git |@@ |--- a/|\+\+\+ b/)")

REPAIR_PLAN_SYSTEM = (
    "Return ONLY one JSON object containing exactly these keys: root_cause, "
    "target_files, target_symbols, required_behavior, regression_test_strategy, "
    "rejected_approaches. Identify a bounded implementation plan from the supplied "
    "EscalationPacket. target_files must be repository-relative paths and "
    "target_symbols must use exact dotted names when applicable."
)

VERIFIED_EDIT_SYSTEM = (
    "Return ONLY one JSON object with key edits. Each item must contain file_path, "
    "node_target (string or null), search, replace, and intent. Use only a target "
    "file and evidence shown in the user payload. Prefer one unique node_target; "
    "otherwise copy a search string verbatim from that file's target evidence. "
    "Do not return a unified diff or infer unseen file content."
)


class RepairContextError(ValueError):
    """The requested context or model-authored anchor failed closed."""


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


def _safe_target(root: Path, relative: str) -> tuple[Path, bool]:
    path = canonical_repo_path(relative)
    if is_sensitive_repo_path(path):
        raise RepairContextError("target file is a sensitive repository path")
    candidate = root / path
    if candidate.is_symlink():
        raise RepairContextError("target file escapes the exact checkout")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise RepairContextError("target file escapes the exact checkout")
    if resolved.exists():
        if not resolved.is_file():
            raise RepairContextError("target file is not a regular file")
        tracked = _git(root, "ls-files", "--error-unmatch", "--", path)
        if tracked.returncode != 0:
            raise RepairContextError("target file is not tracked at the exact checkout")
        return resolved, False
    if resolved.suffix.lower() not in _TEXT_SOURCE_SUFFIXES:
        raise RepairContextError("target file is missing or not an intentional new text file")
    return resolved, True


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RepairContextError("target file must be readable UTF-8 text") from exc


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
                        start_line = min(start_line, *(item.lineno for item in decorators))
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


def build_target_context(state: AgentState, plan: RepairPlan) -> list[Evidence]:
    """Resolve every plan target to bounded exact text in the prepared checkout."""
    if len(plan.target_files) > TARGET_CONTEXT_FILE_LIMIT:
        raise RepairContextError("too many target files for context budget")
    if len(plan.target_symbols) > TARGET_CONTEXT_SYMBOL_LIMIT:
        raise RepairContextError("too many target symbols for context budget")
    root = _validate_checkout(state)
    targets: dict[str, tuple[Path, str | None]] = {}
    for relative in plan.target_files:
        path, is_new = _safe_target(root, relative)
        targets[relative] = (path, None if is_new else _read_utf8(path))

    pending: list[dict[str, str | None]] = []
    represented_files: set[str] = set()
    for symbol in plan.target_symbols:
        matches: list[tuple[str, str, tuple[int, int] | None]] = []
        for relative, (path, source) in targets.items():
            if source is None:
                continue
            if path.suffix.lower() == ".py":
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
        content = _definition_window(source, span) if span else _fallback_window(source, symbol)
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

    for relative, (_path, source) in targets.items():
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
        _EVALUATOR_FIELD_RE.search(str(item.get("content") or ""))
        for item in pending
    ):
        raise RepairContextError("target context contains an evaluator-only field")

    temporary = state.model_copy(deep=True)
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
    state.evidence = temporary.evidence
    return result


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


async def _call_schema(
    state: AgentState,
    *,
    system: str,
    user: str,
    schema: type[SchemaT],
) -> SchemaT:
    model = state.active_model
    provider = state.active_provider
    started = time.monotonic()
    response_text = ""
    try:
        raw = await llm_call(system, user, model=model, provider=provider)
        response_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        result = schema.model_validate(raw)
    except Exception as exc:
        elapsed = time.monotonic() - started
        record_model_invocation(
            state,
            model=model,
            provider=provider,
            node="plan_fix",
            elapsed_seconds=elapsed,
            input_tokens=_estimate_tokens(system, user),
            output_tokens=_estimate_tokens(response_text) if response_text else 0,
            status=("invalid_response" if isinstance(exc, (ValidationError, ValueError)) else "error"),
            error=exc,
        )
        state.token_usage += _estimate_tokens(system, user, response_text)
        raise
    elapsed = time.monotonic() - started
    record_model_invocation(
        state,
        model=model,
        provider=provider,
        node="plan_fix",
        elapsed_seconds=elapsed,
        input_tokens=_estimate_tokens(system, user),
        output_tokens=_estimate_tokens(response_text),
        status="ok",
    )
    state.token_usage += _estimate_tokens(system, user, response_text)
    return result


def _validate_batch(
    state: AgentState,
    plan: RepairPlan,
    batch: VerifiedEditBatch,
    evidence: list[Evidence],
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
        path, is_new = _safe_target(root, edit.file_path)
        if _UNIFIED_DIFF_RE.search(edit.search) or _UNIFIED_DIFF_RE.search(edit.replace):
            raise RepairContextError("unified diff content is not a verified edit anchor")
        if redact_secrets(edit.search) != edit.search or redact_secrets(edit.replace) != edit.replace:
            raise RepairContextError("verified edit contains credential-shaped text")
        if is_new:
            if edit.node_target or edit.search:
                raise RepairContextError("intentional new text file cannot use an existing anchor")
            continue
        anchor = (edit.file_path, edit.node_target or "", edit.search)
        if anchor in anchors:
            raise RepairContextError("verified edit anchors must be unique")
        anchors.add(anchor)
        source = _read_utf8(path)
        file_evidence = evidence_by_file.get(edit.file_path, [])
        if edit.node_target:
            if edit.node_target not in plan.target_symbols:
                raise RepairContextError("verified edit node is outside RepairPlan symbols")
            if locate_node_span(source, edit.node_target) is None:
                raise RepairContextError("verified edit node target is missing or ambiguous")
            if not any(item.symbol == edit.node_target for item in file_evidence):
                raise RepairContextError("verified edit node lacks exact target evidence")
            try:
                replacement = ast.parse(textwrap.dedent(edit.replace))
            except SyntaxError as exc:
                raise RepairContextError("node replacement is not valid Python") from exc
            if len(replacement.body) != 1 or not isinstance(
                replacement.body[0],
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                raise RepairContextError("node replacement must be one complete definition")
            definition = replacement.body[0]
            if definition.name != edit.node_target.rsplit(".", 1)[-1]:
                raise RepairContextError("node replacement does not preserve its target symbol")
        else:
            if not edit.search:
                raise RepairContextError("existing-file verified edit requires an exact anchor")
            if source.count(edit.search) != 1:
                raise RepairContextError("verified edit search anchor is missing or not unique")
            if not any(edit.search in item.content for item in file_evidence):
                raise RepairContextError("verified edit search anchor was not in target evidence")


def verified_edits_to_patch_edits(batch: VerifiedEditBatch) -> list[PatchEdit]:
    """Convert already-validated existing-file edits to the legacy executor type."""
    edits: list[PatchEdit] = []
    for edit in batch.edits:
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
            )
        )
    return edits


async def generate_opus_repair(
    state: AgentState,
    packet: EscalationPacket,
) -> tuple[RepairPlan, VerifiedEditBatch]:
    """Generate patch-free intent, resolve exact context, then request edits."""
    if state.active_provider != "escalation" or not state.escalated:
        raise RepairContextError("two-stage repair requires one-way escalation")
    if packet.base_commit != state.repo_ref:
        raise RepairContextError("EscalationPacket base commit does not match checkout")
    rendered_packet = render_escalation_packet(packet)
    plan = await _call_schema(
        state,
        system=REPAIR_PLAN_SYSTEM,
        user=rendered_packet,
        schema=RepairPlan,
    )
    plan = _safe_plan(plan)
    evidence = build_target_context(state, plan)
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
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(user) > REPAIR_PROMPT_LIMIT:
        raise RepairContextError("verified edit prompt exceeds strict context budget")
    batch = await _call_schema(
        state,
        system=VERIFIED_EDIT_SYSTEM,
        user=user,
        schema=VerifiedEditBatch,
    )
    _validate_batch(state, plan, batch, evidence)
    return plan, batch
