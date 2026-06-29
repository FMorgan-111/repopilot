"""
RepoPilot Eval Report Generator — read eval_results.json and produce summary + markdown.

Reads eval/harness.py output (eval_results.json) and generates:
  1. Terminal summary (to stdout)
  2. eval_summary.md (markdown report)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "eval" / "eval_results.json"
SUMMARY_PATH = REPO_ROOT / "eval" / "eval_summary.md"


def load_results() -> list[dict]:
    """Load eval results from JSON."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from eval results."""
    total = len(results)
    legacy_results = [r for r in results if not _is_agent_v2_result(r)]
    agent_v2_results = [r for r in results if _is_agent_v2_result(r)]
    errors = [r for r in legacy_results if r.get("error")]
    clean = [r for r in legacy_results if not r.get("error")]

    # file_recall@k
    k1_vals = [r["file_recall"]["k1"] for r in clean]
    k3_vals = [r["file_recall"]["k3"] for r in clean]
    k5_vals = [r["file_recall"]["k5"] for r in clean]

    # patch_apply_rate (only among clean runs)
    patch_applies = [r for r in clean if r.get("patch_apply")]
    patch_apply_rate = len(patch_applies) / len(clean) if clean else 0.0

    # test_pass_rate (only among has_tests_changed + patch applied)
    test_runs = [r for r in clean if r.get("has_tests_changed") and r.get("patch_apply")]
    test_passes = [r for r in test_runs if r.get("test_pass")]
    test_pass_rate = len(test_passes) / len(test_runs) if test_runs else None

    # avg_cost
    costs = [r["token_usage"]["cost"] for r in clean if r.get("token_usage", {}).get("cost")]
    avg_cost = statistics.mean(costs) if costs else 0.0
    total_cost = sum(costs) if costs else 0.0
    total_input = sum(r["token_usage"]["input"] for r in clean)
    total_output = sum(r["token_usage"]["output"] for r in clean)

    agent_v2_successes = len([r for r in agent_v2_results if r.get("success")])
    agent_v2_samples = len(agent_v2_results)

    return {
        "total_samples": total,
        "clean_runs": len(clean),
        "errors": len(errors),
        "error_ids": [e["id"] for e in errors],
        "file_recall": {
            "k1_mean": statistics.mean(k1_vals) if k1_vals else 0.0,
            "k1_median": statistics.median(k1_vals) if k1_vals else 0.0,
            "k3_mean": statistics.mean(k3_vals) if k3_vals else 0.0,
            "k3_median": statistics.median(k3_vals) if k3_vals else 0.0,
            "k5_mean": statistics.mean(k5_vals) if k5_vals else 0.0,
            "k5_median": statistics.median(k5_vals) if k5_vals else 0.0,
        },
        "patch_apply_rate": patch_apply_rate,
        "test_pass_rate": test_pass_rate,
        "cost": {
            "avg_per_sample": avg_cost,
            "total": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        },
        "agent_v2": {
            "samples": agent_v2_samples,
            "successes": agent_v2_successes,
            "success_rate": (
                agent_v2_successes / agent_v2_samples
                if agent_v2_samples
                else 0.0
            ),
            "waiting_for_user": len(
                [r for r in agent_v2_results if r.get("waiting_for_user")]
            ),
            "failures": len([r for r in agent_v2_results if not r.get("success")]),
        },
    }


def generate_markdown(results: list[dict], metrics: dict) -> str:
    """Generate eval summary markdown."""
    legacy_results = [r for r in results if not _is_agent_v2_result(r)]
    agent_v2_results = [r for r in results if _is_agent_v2_result(r)]
    lines: list[str] = []
    lines.append("# RepoPilot Eval Summary\n")
    lines.append(f"**Date**: {Path(RESULTS_PATH).stat().st_mtime if RESULTS_PATH.exists() else 'N/A'}\n")
    lines.append(f"**Samples evaluated**: {metrics['total_samples']}\n")
    lines.append(f"**Errors**: {metrics['errors']} ({', '.join(metrics['error_ids']) if metrics['error_ids'] else 'none'})\n")

    lines.append("\n## Aggregate Metrics\n\n")
    lines.append("| Metric | Value |\n")
    lines.append("|--------|-------|\n")
    fr = metrics["file_recall"]
    lines.append(f"| file_recall@1 (mean) | {fr['k1_mean']:.3f} |\n")
    lines.append(f"| file_recall@1 (median) | {fr['k1_median']:.3f} |\n")
    lines.append(f"| file_recall@3 (mean) | {fr['k3_mean']:.3f} |\n")
    lines.append(f"| file_recall@3 (median) | {fr['k3_median']:.3f} |\n")
    lines.append(f"| file_recall@5 (mean) | {fr['k5_mean']:.3f} |\n")
    lines.append(f"| file_recall@5 (median) | {fr['k5_median']:.3f} |\n")
    lines.append(f"| patch_apply_rate | {metrics['patch_apply_rate']:.3f} |\n")

    tpr = metrics["test_pass_rate"]
    if tpr is not None:
        lines.append(f"| test_pass_rate | {tpr:.3f} |\n")
    else:
        lines.append("| test_pass_rate | N/A (no test-relevant samples) |\n")

    c = metrics["cost"]
    lines.append(f"| avg cost per sample | ${c['avg_per_sample']:.6f} |\n")
    lines.append(f"| total cost | ${c['total']:.6f} |\n")
    lines.append(f"| total input tokens | {c['total_input_tokens']:,} |\n")
    lines.append(f"| total output tokens | {c['total_output_tokens']:,} |\n")
    agent_v2 = metrics.get("agent_v2", {})
    if agent_v2.get("samples"):
        lines.append(f"| agent_v2_samples | {agent_v2['samples']} |\n")
        lines.append(f"| agent_v2_success_rate | {agent_v2['success_rate']:.3f} |\n")
        lines.append(f"| agent_v2_waiting_for_user | {agent_v2['waiting_for_user']} |\n")

    lines.append("\n## Per-Sample Results\n\n")
    lines.append("| # | Sample ID | file_recall@1 | file_recall@3 | file_recall@5 | patch_apply | test_pass | cost |\n")
    lines.append("|---|-----------|--------------|--------------|--------------|-------------|-----------|------|\n")
    for i, r in enumerate(legacy_results):
        e = r.get("error")
        if e:
            lines.append(f"| {i+1} | `{r['id'][:60]}` | — | — | — | — | — | error: {e} |\n")
            continue
        fr = r["file_recall"]
        pa = "✓" if r["patch_apply"] else "✗"
        tp_value = r.get("test_pass")
        if tp_value is None:
            tp = "N/A"
        elif tp_value:
            tp = "✓"
        else:
            tp = "✗"
        cost_val = r.get("token_usage", {}).get("cost", 0)
        lines.append(
            f"| {i+1} | `{r['id'][:60]}` "
            f"| {fr['k1']:.2f} | {fr['k3']:.2f} | {fr['k5']:.2f} "
            f"| {pa} | {tp} | ${cost_val:.6f} |\n"
        )

    if agent_v2_results:
        _append_agent_v2_results(lines, agent_v2_results)
        _append_replay_diagnostics(lines, agent_v2_results)

    lines.append("\n## Notes\n\n")
    lines.append("- **file_recall@k**: fraction of actual changed files found in agent's top-k predictions\n")
    lines.append("- **patch_apply_rate**: fraction of agent-generated patches that cleanly apply with `git apply`\n")
    lines.append("- **test_pass_rate**: fraction of applied patches where `pytest` passes (only for `has_tests_changed=true` samples)\n")
    lines.append("- **cost**: estimated DeepSeek API cost ($0.27/M input, $0.36/M output)\n")
    lines.append("- **model**: deepseek-v4-flash (fallback during peak hours)\n")

    return "".join(lines)


def _is_agent_v2_result(result: dict[str, Any]) -> bool:
    return result.get("mode") == "agent_v2"


def _append_agent_v2_results(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.append("\n## Agent V2 Results\n\n")
    lines.append("| Sample ID | Run ID | Final Phase | Waiting | Turns | Tokens | Error |\n")
    lines.append("|-----------|--------|-------------|---------|-------|--------|-------|\n")
    for result in results:
        waiting = "yes" if result.get("waiting_for_user") else "no"
        error = result.get("error") or ""
        error = _markdown_table_cell(error)
        lines.append(
            f"| `{result.get('id', '')[:60]}` "
            f"| `{result.get('run_id', '')}` "
            f"| {result.get('final_phase', '')} "
            f"| {waiting} "
            f"| {result.get('turns_taken', 0)} "
            f"| {result.get('token_used', 0)} "
            f"| {error} |\n"
        )


def _append_replay_diagnostics(
    lines: list[str], results: list[dict[str, Any]]
) -> None:
    lines.append("\n## Replay Diagnostics\n\n")
    for result in results:
        run_id = result.get("run_id", "")
        lines.append(f"### {result.get('id', '')} (`{run_id}`)\n\n")
        replay = result.get("replay")
        if not replay:
            lines.append(f"- Replay unavailable: {result.get('replay_error') or 'missing'}\n")
            continue

        lines.append(f"- Final phase: {replay.get('current_phase', '')}\n")
        latest_frame = _latest_decision_frame(replay)
        if latest_frame is None:
            lines.append("- Latest frame: none\n")
        else:
            stage = latest_frame.get("stage", "")
            frame_id = latest_frame.get("frame_id", "")
            lines.append(f"- Latest frame: {stage} `{frame_id}`\n")
            if latest_frame.get("selected_hypothesis_id"):
                lines.append(
                    f"- Selected hypothesis: {latest_frame['selected_hypothesis_id']}\n"
                )
            selected = latest_frame.get("selected_hypothesis") or {}
            if selected.get("claim"):
                lines.append(f"- Hypothesis claim: {selected['claim']}\n")
            if latest_frame.get("recommended_action"):
                lines.append(
                    f"- Recommended action: {latest_frame['recommended_action']}\n"
                )
            route = latest_frame.get("route") or {}
            if route.get("route"):
                lines.append(f"- Actual route: {route['route']}\n")
            for warning in latest_frame.get("warnings", []):
                lines.append(
                    "- Warning: "
                    f"expected {warning.get('expected_phase', '')} "
                    f"but actual {warning.get('actual_phase', '')}\n"
                )
            for check in latest_frame.get("next_checks", []):
                lines.append(f"- Next check: {check}\n")
        for summary in _diagnostic_summary(replay):
            lines.append(f"- Diagnostic summary: {summary}\n")
        _append_node_diagnostics(lines, replay)
        lines.append("\n")


def _latest_decision_frame(replay: dict[str, Any]) -> dict[str, Any] | None:
    for item in reversed(replay.get("timeline", [])):
        if item.get("type") == "decision_frame":
            return item
    return None


def _diagnostic_summary(replay: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    for item in replay.get("timeline", []):
        if item.get("type") != "node_diagnostic":
            continue
        diagnostic = item.get("diagnostic") or item
        if (
            diagnostic.get("node") == "plan_fix"
            and diagnostic.get("event") == "phase"
            and diagnostic.get("status") == "timeout"
        ):
            timeout = diagnostic.get("phase_timeout_seconds", "")
            summaries.append(f"Planner timeout: plan_fix exceeded {timeout}s.")
    return summaries


def _append_node_diagnostics(lines: list[str], replay: dict[str, Any]) -> None:
    diagnostics = [
        item
        for item in replay.get("timeline", [])
        if item.get("type") == "node_diagnostic"
    ]
    if not diagnostics:
        return

    lines.append("\n#### Node Diagnostics\n\n")
    lines.append("| Node | Event | Status | Error Type | Error |\n")
    lines.append("|------|-------|--------|------------|-------|\n")
    for item in diagnostics:
        diagnostic = item.get("diagnostic") or item
        node = _markdown_table_cell(diagnostic.get("node", ""))
        event = _markdown_table_cell(diagnostic.get("event", ""))
        status = _markdown_table_cell(diagnostic.get("status", ""))
        error_type = _markdown_table_cell(diagnostic.get("error_type", ""))
        error = _markdown_table_cell(diagnostic.get("error", ""))
        lines.append(
            f"| `{node}` | {event} | {status} | {error_type} | {error} |\n"
        )


def _markdown_table_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\r", " ")
    text = text.replace("\n", "<br>")
    return text


def print_summary(metrics: dict) -> None:
    """Print aggregate metrics to terminal."""
    fr = metrics["file_recall"]
    c = metrics["cost"]
    print(f"\n{'='*60}")
    print("REPOPILOT EVAL RESULTS")
    print(f"{'='*60}")
    print(f"Samples:      {metrics['total_samples']} total, {metrics['clean_runs']} clean, {metrics['errors']} errors")
    print(f"file_recall@1: mean={fr['k1_mean']:.3f}  median={fr['k1_median']:.3f}")
    print(f"file_recall@3: mean={fr['k3_mean']:.3f}  median={fr['k3_median']:.3f}")
    print(f"file_recall@5: mean={fr['k5_mean']:.3f}  median={fr['k5_median']:.3f}")
    print(f"patch_apply:  {metrics['patch_apply_rate']:.3f}")
    tpr = metrics["test_pass_rate"]
    if tpr is not None:
        print(f"test_pass:    {tpr:.3f}")
    else:
        print("test_pass:    N/A")
    print(f"avg cost:     ${c['avg_per_sample']:.6f}")
    print(f"total cost:   ${c['total']:.6f}")
    print(f"total tokens: {c['total_input_tokens']:,} in / {c['total_output_tokens']:,} out")
    agent_v2 = metrics.get("agent_v2", {})
    if agent_v2.get("samples"):
        print(f"agent_v2:     {agent_v2['successes']}/{agent_v2['samples']} success")
        print(f"agent_v2 wait:{agent_v2['waiting_for_user']}")
    print(f"{'='*60}")


def main() -> None:
    """Load results, compute metrics, print summary, write markdown."""
    try:
        results = load_results()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run `python eval/harness.py` first to generate eval_results.json")
        return

    metrics = compute_metrics(results)

    print_summary(metrics)

    md = generate_markdown(results, metrics)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"\nMarkdown report saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
