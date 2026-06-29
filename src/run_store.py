"""Persistent storage for paused RepoPilot runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import AgentState


def default_runs_dir() -> Path:
    if repopilot_home := os.getenv("REPOPILOT_HOME"):
        return Path(repopilot_home).expanduser()
    return Path.home() / ".repopilot"


def runs_dir(root_dir: Path | str | None = None) -> Path:
    base_dir = Path(root_dir) if root_dir is not None else default_runs_dir()
    return base_dir / "runs"


def run_path(run_id: str, root_dir: Path | str | None = None) -> Path:
    return runs_dir(root_dir=root_dir) / f"{run_id}.json"


def save_run(state: AgentState, root_dir: Path | str | None = None) -> Path:
    if not state.trace_id:
        raise ValueError("Cannot save a run without a trace_id.")

    path = run_path(state.trace_id, root_dir=root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_run(run_id: str, root_dir: Path | str | None = None) -> AgentState:
    path = run_path(run_id, root_dir=root_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    return AgentState.model_validate(data)


def inspect_run(run_id: str, root_dir: Path | str | None = None) -> dict[str, Any]:
    path = run_path(run_id, root_dir=root_dir)
    return summarize_run(load_run(run_id, root_dir=root_dir), path=path)


def replay_run(run_id: str, root_dir: Path | str | None = None) -> dict[str, Any]:
    return summarize_replay(load_run(run_id, root_dir=root_dir))


def list_runs(root_dir: Path | str | None = None) -> list[dict[str, Any]]:
    directory = runs_dir(root_dir=root_dir)
    if not directory.exists():
        return []
    summaries = [
        summarize_run(load_run(path.stem, root_dir=root_dir), path=path)
        for path in sorted(directory.glob("*.json"), key=lambda item: item.stem)
    ]
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
        "pause": {
            "pending_human_input": state.pending_human_input,
            "question": state.human_input_request.get("question", ""),
            "request": state.human_input_request,
        },
        "timeline": timeline,
    }


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
