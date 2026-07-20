"""Safe execution of policy-approved local tool intents."""

from __future__ import annotations

import hashlib
import shlex
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel

from .evidence import EvidenceCapacityError, EvidenceStore
from .local_search import (
    find_local_references,
    list_local_related_tests,
    read_local_range,
    read_local_symbol,
    is_sensitive_repo_path,
    search_local_symbol,
    search_local_text,
)
from .model_provider import redact_secrets
from .safe_subprocess import (
    BoundedProcessResult,
    SandboxPaths,
    run_bounded_process,
    trusted_executable,
)
from .state import AgentState, ToolAction, ToolInvocation
from .tool_policy import ToolIntent, ToolPolicy


class ToolRouteResult(BaseModel):
    action: ToolAction
    status: Literal["rejected", "ok", "error", "duplicate"]
    args_fingerprint: str
    evidence_id: str | None = None
    error_class: str = ""
    made_progress: bool = False
    control_action: str = ""


def _completed_content(result: BoundedProcessResult) -> str:
    return redact_secrets(
        f"returncode: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )[:8_000]


def _run(
    argv: list[str],
    root: Path,
    *,
    input_text: str | None = None,
    sandbox: SandboxPaths | None = None,
) -> BoundedProcessResult:
    return run_bounded_process(
        argv,
        cwd=root,
        input_text=input_text,
        timeout=300,
        max_output_bytes=8_000,
        isolate_network=sandbox is not None,
        sandbox=sandbox,
    )


def _approved_patch(state: AgentState) -> str:
    approval = state.tool_patch_approval
    if approval is None:
        return ""
    if approval.base_ref.lower() != state.repo_ref.lower():
        raise ValueError("approved patch base mismatch")
    fingerprint = hashlib.sha256(state.patch_content.encode("utf-8")).hexdigest()
    if fingerprint != approval.patch_sha256:
        raise ValueError("approved patch content mismatch")
    for line in state.patch_content.replace("\r\n", "\n").splitlines():
        if line.startswith("diff --git "):
            try:
                header = shlex.split(line)
            except ValueError as exc:
                raise ValueError("approved patch has an invalid header") from exc
            for raw_path in header[2:4]:
                path = raw_path[2:] if raw_path.startswith(("a/", "b/")) else raw_path
                if is_sensitive_repo_path(path):
                    raise ValueError("approved patch contains a sensitive path")
        if line.startswith(("--- a/", "+++ b/")) and is_sensitive_repo_path(line[6:]):
            raise ValueError("approved patch contains a sensitive path")
    return state.patch_content


def _extract_safe_archive(archive: Path, workspace: Path) -> None:
    total_size = 0
    with tarfile.open(archive, mode="r:") as bundle:
        for member in bundle:
            raw = member.name.replace("\\", "/")
            parts = Path(raw).parts
            if not raw or Path(raw).is_absolute() or ".." in parts:
                raise ValueError("unsafe archive path")
            if is_sensitive_repo_path(raw):
                continue
            destination = (workspace / raw).resolve()
            if not destination.is_relative_to(workspace):
                raise ValueError("archive path escaped snapshot")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o700)
                continue
            if not member.isfile():
                continue
            total_size += member.size
            if total_size > 512_000_000:
                raise ValueError("repository snapshot too large")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("archive member missing content")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                while chunk := source.read(64 * 1024):
                    output.write(chunk)
            destination.chmod(0o700 if member.mode & 0o111 else 0o600)


def _validate_generated_snapshot(state: AgentState, workspace: Path) -> None:
    approval = state.tool_patch_approval
    for generated in state.generated_test_approvals:
        if approval is None or generated.patch_gate_fingerprint != approval.patch_gate_fingerprint:
            raise ValueError("generated test lacks matching PatchGate approval")
        path = (workspace / generated.path).resolve()
        if not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError("approved generated test missing from snapshot")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != generated.content_sha256:
            raise ValueError("approved generated test content mismatch")


@contextmanager
def _disposable_test_snapshot(state: AgentState) -> Iterator[SandboxPaths]:
    with tempfile.TemporaryDirectory(prefix="repopilot-tool-sandbox-") as directory:
        sandbox = SandboxPaths.create(Path(directory) / "private")
        archive = sandbox.root / "base.tar"
        git = trusted_executable("git")
        archived = run_bounded_process(
            [git, "-C", state.repo_path, "archive", "--format=tar", f"--output={archive}", state.repo_ref],
            cwd=state.repo_path,
            timeout=60,
            max_output_bytes=8_000,
        )
        if archived.returncode != 0 or not archive.is_file():
            raise RuntimeError("exact-base snapshot creation failed")
        _extract_safe_archive(archive, sandbox.workspace)
        patch = _approved_patch(state)
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
        _validate_generated_snapshot(state, sandbox.workspace)
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


def _safe_changed_paths(state: AgentState, root: Path) -> list[str]:
    accepted: list[str] = []
    for group in _changed_path_groups(state, root):
        if any(is_sensitive_repo_path(path) for path in group):
            continue
        for path in group:
            if path not in accepted:
                accepted.append(path)
    return accepted


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
            content = _completed_content(
                _run(argv, sandbox.workspace, sandbox=sandbox)
            )
    elif action == "inspect_git_diff":
        safe_paths = _safe_changed_paths(state, root)
        if not safe_paths:
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
