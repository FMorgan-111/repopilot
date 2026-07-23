"""Persistent storage for paused RepoPilot runs."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator_safety import contains_evaluator_only
from .home import repopilot_home
from .state import AgentState, Phase
from .summary_safety import sanitize_summary_text

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_RUN_FILE_BYTES = 8 * 1024 * 1024


class ResumeConflictError(RuntimeError):
    """Raised when a durable run no longer matches a resume request."""

    def __init__(self) -> None:
        super().__init__("Run resume conflict.")


def default_runs_dir() -> Path:
    return repopilot_home()


def runs_dir(root_dir: Path | str | None = None) -> Path:
    base_dir = Path(root_dir) if root_dir is not None else default_runs_dir()
    return base_dir / "runs"


def run_path(run_id: str, root_dir: Path | str | None = None) -> Path:
    _validate_run_id(run_id)
    return runs_dir(root_dir=root_dir) / f"{run_id}.json"


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    return run_id


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _open_anchored_directory(path: Path, *, create: bool) -> int:
    """Open a directory by walking no-follow descriptors from its anchor."""
    absolute = path.expanduser().absolute()
    if not absolute.anchor or len(absolute.parts) < 2:
        raise ValueError("run-store root directory is too broad")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
                before = os.stat(
                    part, dir_fd=descriptor, follow_symlinks=False
                )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "run-store root directory components must be real directories"
                )
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(
                    "run-store root directory could not be opened safely"
                ) from exc
            try:
                opened = os.fstat(child)
                after = os.stat(
                    part, dir_fd=descriptor, follow_symlinks=False
                )
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _identity(before) != _identity(opened)
                    or _identity(after) != _identity(opened)
                ):
                    raise ValueError(
                        "run-store root directory changed while it was opened"
                    )
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child

        opened = os.fstat(descriptor)
        path_info = os.stat(absolute, follow_symlinks=False)
        resolved_info = os.stat(
            absolute.resolve(strict=True), follow_symlinks=False
        )
        if (
            _identity(path_info) != _identity(opened)
            or _identity(resolved_info) != _identity(opened)
        ):
            raise ValueError("run-store root directory changed while it was opened")
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _open_runs_directory(
    root_dir: Path | str | None,
    *,
    create: bool,
) -> Iterator[tuple[Path, int, int]]:
    root = Path(root_dir) if root_dir is not None else default_runs_dir()
    directory = root / "runs"
    root_descriptor = -1
    runs_descriptor = -1
    try:
        root_descriptor = _open_anchored_directory(root, create=create)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

        try:
            runs_before = os.stat(
                "runs", dir_fd=root_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir("runs", 0o700, dir_fd=root_descriptor)
            except FileExistsError:
                pass
            else:
                os.fsync(root_descriptor)
            runs_before = os.stat(
                "runs", dir_fd=root_descriptor, follow_symlinks=False
            )
        if stat.S_ISLNK(runs_before.st_mode) or not stat.S_ISDIR(runs_before.st_mode):
            raise ValueError("runs directory must be a real directory, not a symlink")
        try:
            runs_descriptor = os.open(
                "runs", flags, dir_fd=root_descriptor
            )
        except OSError as exc:
            raise ValueError("runs directory could not be opened safely") from exc
        runs_opened = os.fstat(runs_descriptor)
        runs_after = os.stat(
            "runs", dir_fd=root_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(runs_opened.st_mode)
            or _identity(runs_before) != _identity(runs_opened)
            or _identity(runs_after) != _identity(runs_opened)
        ):
            raise ValueError("runs directory changed while it was opened")
        yield directory, root_descriptor, runs_descriptor
    finally:
        if runs_descriptor >= 0:
            os.close(runs_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)


def _read_run_bytes_at(
    run_id: str, directory_fd: int
) -> tuple[bytes, os.stat_result]:
    _validate_run_id(run_id)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(
            f"{run_id}.json",
            flags,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError("run file must be a regular file") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("run file must be a regular file")
        if opened.st_size > MAX_RUN_FILE_BYTES:
            raise ValueError("run file exceeds the size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_RUN_FILE_BYTES + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_RUN_FILE_BYTES:
                raise ValueError("run file exceeds the size limit")
            chunks.append(chunk)
        return b"".join(chunks), opened
    finally:
        os.close(descriptor)


def _read_run_bytes(run_id: str, root_dir: Path | str | None) -> bytes:
    _validate_run_id(run_id)
    with _open_runs_directory(root_dir, create=False) as (_, _, directory_fd):
        content, _opened = _read_run_bytes_at(run_id, directory_fd)
        return content


def _atomic_write_run_at(
    run_id: str,
    payload: bytes,
    directory_fd: int,
    path: Path,
    *,
    rollback_payload: bytes | None = None,
) -> Path:
    if len(payload) > MAX_RUN_FILE_BYTES:
        raise ValueError("run file exceeds the size limit")
    temp_name = ""
    descriptor = -1
    replaced = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        for _ in range(16):
            temp_name = f".{run_id}.{secrets.token_hex(12)}.tmp"
            try:
                descriptor = os.open(
                    temp_name,
                    flags,
                    0o600,
                    dir_fd=directory_fd,
                )
                break
            except FileExistsError:
                continue
        else:
            raise OSError("could not allocate a unique run-store temporary file")
        try:
            os.fchmod(descriptor, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while persisting run")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            descriptor = -1
        os.replace(
            temp_name,
            f"{run_id}.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temp_name = ""
        replaced = True
        os.fsync(directory_fd)
        return path
    except BaseException:
        if replaced and rollback_payload is not None:
            try:
                _atomic_write_run_at(
                    run_id,
                    rollback_payload,
                    directory_fd,
                    path,
                )
            except BaseException:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temp_name:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _atomic_write_run(
    run_id: str,
    payload: bytes,
    root_dir: Path | str | None,
) -> Path:
    path = run_path(run_id, root_dir=root_dir)
    with _open_runs_directory(root_dir, create=True) as (_, _, directory_fd):
        return _atomic_write_run_at(run_id, payload, directory_fd, path)


def _lock_is_safe(
    info: os.stat_result,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o600
        and (expected_identity is None or _identity(info) == expected_identity)
    )


@contextmanager
def _run_lock(
    run_id: str,
    root_dir: Path | str | None,
) -> Iterator[tuple[Path, int]]:
    """Lock one run while retaining the descriptor that anchors its state."""
    _validate_run_id(run_id)
    lock_name = f".{run_id}.lock"
    with _open_runs_directory(root_dir, create=True) as (
        directory,
        _root_descriptor,
        directory_fd,
    ):
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            lock_fd = os.open(
                lock_name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ValueError("run lock must be a regular file") from exc

        acquired = False
        try:
            opened = os.fstat(lock_fd)
            if not _lock_is_safe(opened):
                raise ValueError("run lock must be a regular file")
            try:
                named = os.stat(
                    lock_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    "run lock must be a regular file"
                ) from exc
            if not _lock_is_safe(
                named,
                expected_identity=_identity(opened),
            ):
                raise ValueError("run lock must be a regular file")

            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            acquired = True

            locked = os.fstat(lock_fd)
            try:
                named_after_lock = os.stat(
                    lock_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    "run lock must be a regular file"
                ) from exc
            if (
                not _lock_is_safe(
                    locked,
                    expected_identity=_identity(opened),
                )
                or not _lock_is_safe(
                    named_after_lock,
                    expected_identity=_identity(opened),
                )
            ):
                raise ValueError("run lock must be a regular file")

            yield directory, directory_fd
        finally:
            try:
                if acquired:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def _encoded_run_state(state: AgentState) -> bytes:
    payload = state.model_dump(mode="json")
    payload = _clear_legacy_evaluator_patch_state(payload)
    payload["attempt_outcome_summary"] = _safe_outcome_summary(state)
    return json.dumps(payload, indent=2).encode("utf-8")


def save_run(
    state: AgentState,
    root_dir: Path | str | None = None,
    *,
    rollback_state: AgentState | None = None,
) -> Path:
    if not state.trace_id:
        raise ValueError("Cannot save a run without a trace_id.")

    _validate_run_id(state.trace_id)
    path = run_path(state.trace_id, root_dir=root_dir)
    with _run_lock(state.trace_id, root_dir) as (_, directory_fd):
        return _atomic_write_run_at(
            state.trace_id,
            _encoded_run_state(state),
            directory_fd,
            path,
            rollback_payload=(
                _encoded_run_state(rollback_state)
                if rollback_state is not None
                else None
            ),
        )


def claim_run_for_resume(
    run_id: str,
    expected_state: AgentState,
    *,
    root_dir: Path | str | None = None,
) -> AgentState:
    """Atomically consume one paused request and return its claimed state."""
    _validate_run_id(run_id)
    if not isinstance(expected_state, AgentState):
        raise ResumeConflictError

    path = run_path(run_id, root_dir=root_dir)
    with _run_lock(run_id, root_dir) as (_, directory_fd):
        try:
            content, _opened = _read_run_bytes_at(run_id, directory_fd)
        except FileNotFoundError:
            raise ResumeConflictError from None
        durable = _state_from_run_bytes(content)
        if (
            durable.trace_id != run_id
            or expected_state.trace_id != run_id
            or durable.model_dump(mode="json")
            != expected_state.model_dump(mode="json")
            or durable.current_phase != Phase.WAITING_FOR_USER
            or not durable.pending_human_input
            or not durable.human_input_request
            or durable.resume_in_progress
        ):
            raise ResumeConflictError

        claimed = durable.model_copy(deep=True)
        claimed.resume_in_progress = True
        claimed.current_phase = Phase.PLAN
        claimed.pending_human_input = False
        claimed.human_input_request = {}
        _atomic_write_run_at(
            run_id,
            _encoded_run_state(claimed),
            directory_fd,
            path,
        )
        return claimed


def load_run(run_id: str, root_dir: Path | str | None = None) -> AgentState:
    return _state_from_run_bytes(_read_run_bytes(run_id, root_dir))


def _state_from_run_bytes(content: bytes) -> AgentState:
    data = json.loads(content.decode("utf-8"))
    return AgentState.model_validate(_clear_legacy_evaluator_patch_state(data))


def _clear_legacy_evaluator_patch_state(data: dict[str, Any]) -> dict[str, Any]:
    """Remove legacy patch-bearing fields before they cross the state boundary."""
    cleaned = dict(data)
    patch_fields = (
        cleaned.get("patch_content"),
        cleaned.get("patch_edits"),
        cleaned.get("active_repair_plan"),
    )
    if any(contains_evaluator_only(item) for item in patch_fields if item):
        cleaned["patch_content"] = ""
        cleaned["patch_edits"] = []
        cleaned["active_repair_plan"] = None
        cleaned["tool_patch_approval"] = None
        cleaned["generated_test_approvals"] = []
        cleaned["coverage_proof"] = None
        cleaned["coverage_status"] = "pending"
    attempts: list[Any] = []
    for raw in cleaned.get("fix_attempts") or []:
        if not isinstance(raw, dict):
            attempts.append(raw)
            continue
        attempt = dict(raw)
        patch_values = (attempt.get("patch_content"), attempt.get("patch_edits"))
        if any(contains_evaluator_only(item) for item in patch_values if item):
            attempt.update(
                {
                    "patch_content": "",
                    "patch_edits": [],
                    "file_path": "",
                    "test_result": "",
                    "failure_kind": "evaluator_contaminated",
                    "error_log": "",
                    "success": False,
                }
            )
        attempts.append(attempt)
    cleaned["fix_attempts"] = attempts
    return cleaned


def inspect_run(run_id: str, root_dir: Path | str | None = None) -> dict[str, Any]:
    _validate_run_id(run_id)
    with _open_runs_directory(root_dir, create=False) as (_, _, directory_fd):
        content, opened = _read_run_bytes_at(run_id, directory_fd)
    summary = summarize_run(_state_from_run_bytes(content))
    summary["updated_at"] = _updated_at_stat(opened)
    return summary


def replay_run(run_id: str, root_dir: Path | str | None = None) -> dict[str, Any]:
    return summarize_replay(load_run(run_id, root_dir=root_dir))


def list_runs(root_dir: Path | str | None = None) -> list[dict[str, Any]]:
    try:
        with _open_runs_directory(root_dir, create=False) as (_, _, directory_fd):
            names = sorted(
                entry.name
                for entry in os.scandir(directory_fd)
                if entry.name.endswith(".json")
                and _RUN_ID_RE.fullmatch(entry.name[:-5])
                and entry.is_file(follow_symlinks=False)
            )
            summaries = []
            for name in names:
                run_id = name[:-5]
                content, opened = _read_run_bytes_at(run_id, directory_fd)
                summary = summarize_run(_state_from_run_bytes(content))
                summary["updated_at"] = _updated_at_stat(opened)
                summaries.append(summary)
    except FileNotFoundError:
        return []
    return summaries


def summarize_run(state: AgentState, path: Path | None = None) -> dict[str, Any]:
    latest_frame = state.decision_frame
    if latest_frame is None and state.frame_history:
        latest_frame = state.frame_history[-1]

    return {
        "run_id": state.trace_id or (path.stem if path else ""),
        "issue_url": state.issue_url,
        "current_phase": state.current_phase.value,
        "pending_human_input": state.pending_human_input,
        "human_input_question": state.human_input_request.get("question", ""),
        "latest_decision_frame": (
            latest_frame.model_dump(mode="json") if latest_frame else None
        ),
        "updated_at": _updated_at(path) if path else "",
    }


def summarize_replay(state: AgentState) -> dict[str, Any]:
    used_route_indexes: set[int] = set()
    timeline: list[dict[str, Any]] = []

    for frame in state.frame_history:
        route_index, route = _route_for_frame(state.route_decisions, frame.frame_id)
        if route_index is not None:
            used_route_indexes.add(route_index)
        timeline.append(
            {
                "index": len(timeline) + 1,
                "type": "decision_frame",
                "frame_id": frame.frame_id,
                "stage": frame.stage,
                "summary": frame.summary,
                "selected_hypothesis_id": frame.selected_hypothesis_id,
                "selected_hypothesis": _selected_hypothesis(frame),
                "recommended_action": frame.recommended_action,
                "risk": frame.risk,
                "confidence": frame.confidence,
                "route": route,
                "warnings": _warnings_for_frame(state.decision_warnings, frame.frame_id),
                "next_checks": frame.next_checks,
                "trace_notes": frame.trace_notes,
            }
        )

    for index, route in enumerate(state.route_decisions):
        if index in used_route_indexes:
            continue
        timeline.append(
            {
                "index": len(timeline) + 1,
                "type": "route_decision",
                "route": route,
            }
        )

    for diagnostic in state.node_diagnostics:
        timeline.append(
            {
                "index": len(timeline) + 1,
                "type": "node_diagnostic",
                "diagnostic": diagnostic,
            }
        )

    return {
        "run_id": state.trace_id,
        "issue_url": state.issue_url,
        "current_phase": state.current_phase.value,
        "attempt_outcome_summary": _safe_outcome_summary(state),
        "summary_token_usage": state.summary_token_usage,
        "pause": {
            "pending_human_input": state.pending_human_input,
            "question": state.human_input_request.get("question", ""),
            "request": state.human_input_request,
        },
        "timeline": timeline,
    }


def _safe_outcome_summary(state: AgentState) -> str:
    return sanitize_summary_text(
        state.attempt_outcome_summary,
        denied_literals=(item.path for item in state.generated_test_approvals),
    )


def format_replay_markdown(replay: dict[str, Any]) -> str:
    lines = [
        f"# RepoPilot Replay: {replay.get('run_id', '')}",
        "",
        f"- Issue: {replay.get('issue_url', '')}",
        f"- Final phase: {replay.get('current_phase', '')}",
    ]
    pause = replay.get("pause") or {}
    lines.append(
        f"- Pending human input: {'yes' if pause.get('pending_human_input') else 'no'}"
    )
    if pause.get("question"):
        lines.append(f"- Question: {pause['question']}")

    lines.extend(["", "## Timeline"])

    for item in replay.get("timeline", []):
        lines.append("")
        if item.get("type") == "decision_frame":
            _append_frame_markdown(lines, item)
        elif item.get("type") == "route_decision":
            _append_route_markdown(lines, item)
        elif item.get("type") == "node_diagnostic":
            _append_node_diagnostic_markdown(lines, item)

    return "\n".join(lines)


def _append_frame_markdown(lines: list[str], item: dict[str, Any]) -> None:
    lines.append(
        f"### {item.get('index')}. {str(item.get('stage', '')).upper()} "
        f"{item.get('frame_id', '')}"
    )
    if item.get("summary"):
        lines.extend(["", item["summary"]])

    lines.append("")
    if item.get("selected_hypothesis_id"):
        lines.append(f"- Selected hypothesis: {item['selected_hypothesis_id']}")
    selected = item.get("selected_hypothesis") or {}
    if selected.get("claim"):
        lines.append(f"- Hypothesis claim: {selected['claim']}")
    if item.get("recommended_action"):
        lines.append(f"- Recommended action: {item['recommended_action']}")
    if item.get("risk"):
        lines.append(f"- Risk: {item['risk']}")
    if item.get("confidence") is not None:
        lines.append(f"- Confidence: {item['confidence']}")

    route = item.get("route") or {}
    if route.get("route"):
        lines.append(f"- Route: {route['route']}")

    for warning in item.get("warnings", []):
        expected = warning.get("expected_phase", "")
        actual = warning.get("actual_phase", "")
        lines.append(f"- Warning: expected {expected} but actual {actual}")

    for check in item.get("next_checks", []):
        lines.append(f"- Next check: {check}")

    if item.get("trace_notes"):
        lines.append(f"- Trace notes: {item['trace_notes']}")


def _append_route_markdown(lines: list[str], item: dict[str, Any]) -> None:
    lines.append(f"### {item.get('index')}. Route Decision")
    lines.append("")
    route = item.get("route") or {}
    if route.get("route"):
        lines.append(f"- Route: {route['route']}")
    if route.get("source"):
        lines.append(f"- Source: {route['source']}")
    if route.get("fallback_reason"):
        lines.append(f"- Fallback reason: {route['fallback_reason']}")


def _append_node_diagnostic_markdown(
    lines: list[str], item: dict[str, Any]
) -> None:
    lines.append(f"### {item.get('index')}. Node Diagnostic")
    lines.append("")
    diagnostic = item.get("diagnostic") or {}
    for key, label in [
        ("node", "Node"),
        ("event", "Event"),
        ("status", "Status"),
        ("elapsed_seconds", "Elapsed seconds"),
        ("error_type", "Error type"),
        ("error", "Error"),
        ("phase_timeout_seconds", "Phase timeout seconds"),
        ("prompt_tokens_estimate", "Prompt tokens estimate"),
        ("response_tokens_estimate", "Response tokens estimate"),
    ]:
        if key in diagnostic:
            lines.append(f"- {label}: {diagnostic[key]}")


def _route_for_frame(
    route_decisions: list[dict[str, Any]], frame_id: str
) -> tuple[int | None, dict[str, Any] | None]:
    for index, route in enumerate(route_decisions):
        if route.get("frame_id") == frame_id:
            return index, route
    return None, None


def _warnings_for_frame(
    decision_warnings: list[dict[str, Any]], frame_id: str
) -> list[dict[str, Any]]:
    return [
        warning
        for warning in decision_warnings
        if warning.get("frame_id") == frame_id
    ]


def _selected_hypothesis(frame: Any) -> dict[str, Any] | None:
    if not frame.selected_hypothesis_id:
        return None
    for hypothesis in frame.hypotheses:
        if hypothesis.id == frame.selected_hypothesis_id:
            return hypothesis.model_dump(mode="json")
    return None


def _updated_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _updated_at_stat(info: os.stat_result) -> str:
    return datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat()
