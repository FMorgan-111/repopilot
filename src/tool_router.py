"""Safe execution of policy-approved local tool intents."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .evidence import EvidenceStore
from .local_search import (
    find_local_references,
    list_local_related_tests,
    read_local_range,
    read_local_symbol,
    search_local_symbol,
    search_local_text,
)
from .model_provider import redact_secrets
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


def _completed_content(result: subprocess.CompletedProcess[str]) -> str:
    return redact_secrets(
        f"returncode: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )[:8_000]


def _run(argv: list[str], root: Path, *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=root,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


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
        content = _completed_content(_run(argv, root))
    elif action == "inspect_git_diff":
        content = _completed_content(
            _run(
                [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--unified=3",
                    state.repo_ref,
                    "--",
                ],
                root,
            )
        )
    elif action == "validate_patch":
        result = _run(
            ["git", "apply", "--check", "--recount", "-"],
            root,
            input_text=state.patch_content,
        )
        if result.returncode != 0:
            result = _run(
                ["git", "apply", "--check", "--reverse", "--recount", "-"],
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
        status = "ok" if added.added else "duplicate"
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
