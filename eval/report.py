"""
RepoPilot Eval Report Generator — read eval_results.json and produce summary + markdown.

Reads eval/harness.py output (eval_results.json) and generates:
  1. Terminal summary (to stdout)
  2. eval_summary.md (markdown report)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .failure_taxonomy import summarize as summarize_failures
else:  # Support `python eval/report.py` and imports from the eval directory.
    from failure_taxonomy import summarize as summarize_failures

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "eval" / "eval_results.json"
SUMMARY_PATH = REPO_ROOT / "eval" / "eval_summary.md"
_VERIFIED_COVERAGE_STATUSES = {"existing_verified", "generated_verified"}
_EVALUATOR_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gold[_ -]?patch|test[_ -]?patch|"
    r"fail[_ -]?to[_ -]?pass|pass[_ -]?to[_ -]?pass)"
    r"(?![A-Za-z0-9])"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[^\s,;]+")
_SK_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}")
_ARCHIVE_PAYLOAD_RE = re.compile(r"(?:PK\\x03\\x04|PK\x03\x04)")

_AGENT_RESULT_DEFAULTS: dict[str, Any] = {
    "models_used": [],
    "escalated": False,
    "escalation_reason": "",
    "model_invocations": [],
    "tool_invocations": [],
    "unique_evidence_count": 0,
    "max_consecutive_no_progress": 0,
    "attempt_outcome_summary": "",
    "coverage_status": "pending",
    "coverage_test_files": [],
    "coverage_test_command": "",
    "coverage_proof": None,
    "coverage_failure_reason": "",
    "test_generation_attempts": 0,
    "official_resolved": None,
}


def _safe_text(value: Any, limit: int = 2_000) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _SK_RE.sub("[REDACTED]", text)
    boundaries = [
        match.start()
        for match in (_EVALUATOR_RE.search(text), _ARCHIVE_PAYLOAD_RE.search(text))
        if match is not None
    ]
    if boundaries:
        text = text[: min(boundaries)]
    return text.strip()[:limit]


def _safe_coverage_run(value: Any) -> dict[str, Any]:
    run = value if isinstance(value, dict) else {}
    return {
        "outcome": _safe_text(run.get("outcome"), 40),
        "failing_test_ids": [
            _safe_text(item, 500)
            for item in (run.get("failing_test_ids") or [])[:100]
            if _safe_text(item, 500)
        ],
        "assertion_fingerprint": _safe_text(
            run.get("assertion_fingerprint"), 128
        ),
    }


def _safe_coverage_proof(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    status = _safe_text(value.get("status"), 40)
    test_files = [
        _safe_text(item, 500)
        for item in (value.get("test_files") or [])[:16]
        if _safe_text(item, 500)
    ]
    if status not in _VERIFIED_COVERAGE_STATUSES or not test_files:
        return None
    proof = {
        "source": _safe_text(value.get("source"), 40),
        "status": status,
        "test_files": test_files,
        "fixed_runs": [
            _safe_coverage_run(item)
            for item in (value.get("fixed_runs") or [])[:2]
        ],
        "base_runs": [
            _safe_coverage_run(item)
            for item in (value.get("base_runs") or [])[:2]
        ],
    }
    fixed_runs = proof["fixed_runs"]
    base_runs = proof["base_runs"]
    if len(fixed_runs) != 2 or len(base_runs) != 2:
        return None
    if any(run["outcome"] != "pass" for run in fixed_runs):
        return None
    if any(
        run["outcome"] != "assertion_failure"
        or not run["failing_test_ids"]
        or not run["assertion_fingerprint"]
        for run in base_runs
    ):
        return None
    if (
        base_runs[0]["failing_test_ids"] != base_runs[1]["failing_test_ids"]
        or base_runs[0]["assertion_fingerprint"]
        != base_runs[1]["assertion_fingerprint"]
    ):
        return None
    return proof


def _normalize_result(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if not _is_agent_v2_result(result):
        return result
    for key, default in _AGENT_RESULT_DEFAULTS.items():
        if key not in result:
            result[key] = list(default) if isinstance(default, list) else default
    result["attempt_outcome_summary"] = _safe_text(
        result.get("attempt_outcome_summary"), 200
    )
    status = result.get("coverage_status")
    if status not in {
        "pending",
        "existing_verified",
        "generated_verified",
        "failed",
    }:
        status = "pending"
    result["coverage_status"] = status
    result["coverage_proof"] = _safe_coverage_proof(result.get("coverage_proof"))
    verified = bool(
        status in _VERIFIED_COVERAGE_STATUSES
        and result["coverage_proof"]
        and result["coverage_proof"]["status"] == status
    )
    claimed_success = result.get("agent_success", result.get("success", False))
    result["agent_success"] = bool(claimed_success and verified)
    result["success"] = result["agent_success"]
    result["error"] = _safe_text(result.get("error")) or None
    return result


def load_results(results_path: Path | str = RESULTS_PATH) -> list[dict]:
    """Load eval results from JSON."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    return [_normalize_result(value) for value in values]


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from eval results."""
    results = [_normalize_result(result) for result in results]
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

    agent_v2_successes = len(
        [r for r in agent_v2_results if r.get("agent_success")]
    )
    agent_v2_samples = len(agent_v2_results)
    by_evaluation_mode = {}
    for evaluation_mode in ("end_to_end", "oracle_files"):
        mode_results = [
            result
            for result in agent_v2_results
            if _evaluation_mode(result) == evaluation_mode
        ]
        mode_successes = len(
            [result for result in mode_results if result.get("agent_success")]
        )
        taxonomy = summarize_failures(mode_results)
        official_results = [
            result
            for result in mode_results
            if result.get("official_resolved") is not None
        ]
        official_resolved = len(
            [result for result in official_results if result.get("official_resolved")]
        )
        by_evaluation_mode[evaluation_mode] = {
            "samples": len(mode_results),
            "agent_successes": mode_successes,
            "agent_success_rate": (
                mode_successes / len(mode_results) if mode_results else 0.0
            ),
            "official_scored_samples": len(official_results),
            "official_resolved": official_resolved,
            "official_resolved_rate": (
                official_resolved / len(official_results)
                if official_results
                else None
            ),
            # Compatibility aliases for API consumers; report labels below use
            # the explicit internal/official names.
            "successes": mode_successes,
            "success_rate": mode_successes / len(mode_results) if mode_results else 0.0,
            "failure_taxonomy": taxonomy["decisive"],
            "models": sorted(
                {
                    safe
                    for result in mode_results
                    if result.get("model")
                    and (safe := _safe_text(result["model"], 300))
                }
            ),
            "commits": sorted(
                {
                    safe
                    for result in mode_results
                    if result.get("commit_sha")
                    and (safe := _safe_text(result["commit_sha"], 300))
                }
            ),
        }

    model_samples: dict[str, set[str]] = {}
    model_invocations: Counter[str] = Counter()
    model_successes: Counter[str] = Counter()
    for index, result in enumerate(agent_v2_results):
        sample_key = str(result.get("id") or index)
        models = list(result.get("models_used") or [])
        for invocation in result.get("model_invocations") or []:
            if not isinstance(invocation, dict):
                continue
            model = _safe_text(invocation.get("model"), 300)
            if model:
                model_invocations[model] += 1
                if model not in models:
                    models.append(model)
        for model in models:
            safe_model = _safe_text(model, 300)
            if not safe_model:
                continue
            model_samples.setdefault(safe_model, set()).add(sample_key)
            if result.get("agent_success"):
                model_successes[safe_model] += 1
    by_model = {
        model: {
            "samples": len(model_samples[model]),
            "invocations": model_invocations[model],
            "agent_successes": model_successes[model],
        }
        for model in sorted(model_samples)
    }
    escalation_reasons = Counter(
        _safe_text(result.get("escalation_reason"), 100)
        for result in agent_v2_results
        if result.get("escalated") and result.get("escalation_reason")
    )
    tool_statuses: Counter[str] = Counter()
    for result in agent_v2_results:
        for invocation in result.get("tool_invocations") or []:
            if isinstance(invocation, dict):
                status = _safe_text(invocation.get("status"), 100)
                if status:
                    tool_statuses[status] += 1
    coverage_statuses = Counter(
        str(result.get("coverage_status") or "pending")
        for result in agent_v2_results
    )
    official_results = [
        result
        for result in agent_v2_results
        if result.get("official_resolved") is not None
    ]
    official_resolved = len(
        [result for result in official_results if result.get("official_resolved")]
    )

    return {
        "total_samples": total,
        "clean_runs": len(clean),
        "errors": len(errors),
        "error_ids": [_safe_text(e["id"], 500) for e in errors],
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
            "agent_successes": agent_v2_successes,
            "agent_success_rate": (
                agent_v2_successes / agent_v2_samples if agent_v2_samples else 0.0
            ),
            "official_scored_samples": len(official_results),
            "official_resolved": official_resolved,
            "official_resolved_rate": (
                official_resolved / len(official_results)
                if official_results
                else None
            ),
            "successes": agent_v2_successes,
            "success_rate": (
                agent_v2_successes / agent_v2_samples
                if agent_v2_samples
                else 0.0
            ),
            "waiting_for_user": len(
                [r for r in agent_v2_results if r.get("waiting_for_user")]
            ),
            "failures": len(
                [r for r in agent_v2_results if not r.get("agent_success")]
            ),
            "by_model": by_model,
            "escalation_reasons": dict(sorted(escalation_reasons.items())),
            "tool_statuses": dict(sorted(tool_statuses.items())),
            "unique_evidence_count": sum(
                max(0, int(result.get("unique_evidence_count") or 0))
                for result in agent_v2_results
            ),
            "max_consecutive_no_progress": max(
                (
                    max(0, int(result.get("max_consecutive_no_progress") or 0))
                    for result in agent_v2_results
                ),
                default=0,
            ),
            "coverage": {
                "statuses": dict(sorted(coverage_statuses.items())),
                "proofs": len(
                    [result for result in agent_v2_results if result.get("coverage_proof")]
                ),
                "test_generation_attempts": sum(
                    max(0, int(result.get("test_generation_attempts") or 0))
                    for result in agent_v2_results
                ),
            },
            "by_evaluation_mode": by_evaluation_mode,
        },
    }


def generate_markdown(results: list[dict], metrics: dict) -> str:
    """Generate eval summary markdown."""
    results = [_normalize_result(result) for result in results]
    legacy_results = [r for r in results if not _is_agent_v2_result(r)]
    agent_v2_results = [r for r in results if _is_agent_v2_result(r)]
    lines: list[str] = []
    lines.append("# RepoPilot Eval Summary\n")
    lines.append(f"**Date**: {Path(RESULTS_PATH).stat().st_mtime if RESULTS_PATH.exists() else 'N/A'}\n")
    lines.append(f"**Samples evaluated**: {metrics['total_samples']}\n")
    lines.append(f"**Errors**: {metrics['errors']} ({', '.join(metrics['error_ids']) if metrics['error_ids'] else 'none'})\n")
    models = sorted(
        {
            safe
            for result in agent_v2_results
            for model in (
                list(result.get("models_used") or [])
                or ([result.get("model")] if result.get("model") else [])
            )
            if model
            and (safe := _safe_text(model, 300))
        }
    )
    commits = sorted(
        {
            safe
            for result in agent_v2_results
            if result.get("commit_sha")
            and (safe := _safe_text(result.get("commit_sha"), 300))
        }
    )
    if models:
        lines.append(f"**Models**: {', '.join(models)}\n")
    if commits:
        lines.append(f"**Commits**: {', '.join(commits)}\n")

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
        lines.append(
            "| agent_v2_combined_all_modes_agent_success_rate "
            f"| {agent_v2['agent_success_rate']:.3f} |\n"
        )
        if agent_v2["official_scored_samples"]:
            lines.append(
                "| official_resolved_rate "
                f"| {agent_v2['official_resolved_rate']:.3f} |\n"
            )
        else:
            lines.append("| official_resolved_rate | N/A (not scored) |\n")
        lines.append(f"| agent_v2_waiting_for_user | {agent_v2['waiting_for_user']} |\n")

        _append_evaluation_mode_summary(lines, agent_v2["by_evaluation_mode"])
        _append_success_first_summary(lines, agent_v2)

    lines.append("\n## Per-Sample Results\n\n")
    lines.append("| # | Sample ID | file_recall@1 | file_recall@3 | file_recall@5 | patch_apply | test_pass | cost |\n")
    lines.append("|---|-----------|--------------|--------------|--------------|-------------|-----------|------|\n")
    for i, r in enumerate(legacy_results):
        e = r.get("error")
        if e:
            lines.append(
                f"| {i+1} | `{_safe_text(r['id'], 60)}` | — | — | — | — | — "
                f"| error: {_markdown_table_cell(e)} |\n"
            )
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
        _append_coverage_proofs(lines, agent_v2_results)
        _append_replay_diagnostics(lines, agent_v2_results)

    lines.append("\n## Notes\n\n")
    lines.append("- **file_recall@k**: fraction of actual changed files found in agent's top-k predictions\n")
    lines.append("- **patch_apply_rate**: fraction of agent-generated patches that cleanly apply with `git apply`\n")
    lines.append("- **test_pass_rate**: fraction of applied patches where `pytest` passes (only for `has_tests_changed=true` samples)\n")
    lines.append("- **cost**: legacy provider estimate; verify against the configured gateway's current billing\n")

    return "".join(lines)


def _is_agent_v2_result(result: dict[str, Any]) -> bool:
    return result.get("mode") == "agent_v2"


def _evaluation_mode(result: dict[str, Any]) -> str:
    mode = result.get("evaluation_mode", "end_to_end")
    return mode if mode in {"end_to_end", "oracle_files"} else "end_to_end"


def _append_evaluation_mode_summary(
    lines: list[str], by_evaluation_mode: dict[str, dict[str, Any]]
) -> None:
    lines.append("\n## Agent V2 Evaluation Modes\n\n")
    lines.append(
        "| Mode | Samples | Models | Commits | Agent Success | Official Resolved | "
        "Decisive Failure Taxonomy |\n"
    )
    lines.append(
        "|------|---------|--------|---------|---------------|-------------------|"
        "---------------------------|\n"
    )
    for mode, mode_metrics in by_evaluation_mode.items():
        samples = mode_metrics["samples"]
        if not samples:
            continue
        models = _markdown_table_cell(", ".join(mode_metrics["models"]) or "none")
        commits = _markdown_table_cell(", ".join(mode_metrics["commits"]) or "none")
        taxonomy = _markdown_table_cell(
            _format_failure_taxonomy(mode_metrics["failure_taxonomy"])
        )
        official = (
            f"{mode_metrics['official_resolved']}/"
            f"{mode_metrics['official_scored_samples']} "
            f"({mode_metrics['official_resolved_rate']:.1%})"
            if mode_metrics["official_scored_samples"]
            else "not scored"
        )
        lines.append(
            f"| {mode} | {samples} | {models} | {commits} "
            f"| {mode_metrics['agent_successes']}/{samples} "
            f"({mode_metrics['agent_success_rate']:.1%}) | {official} "
            f"| {taxonomy} |\n"
        )


def _append_success_first_summary(lines: list[str], agent: dict[str, Any]) -> None:
    lines.append("\n## Success-First Diagnostics\n\n")
    lines.append("| Model | Samples | Invocations | Agent Successes |\n")
    lines.append("|-------|---------|-------------|-----------------|\n")
    for model, values in agent["by_model"].items():
        lines.append(
            f"| {_markdown_table_cell(model)} | {values['samples']} "
            f"| {values['invocations']} | {values['agent_successes']} |\n"
        )
    if not agent["by_model"]:
        lines.append("| none | 0 | 0 | 0 |\n")
    lines.append(
        "\n- Escalation reasons: "
        f"{_format_failure_taxonomy(agent['escalation_reasons'])}\n"
    )
    lines.append(
        "- Tool statuses: "
        f"{_format_failure_taxonomy(agent['tool_statuses'])}\n"
    )
    lines.append(f"- Unique evidence: {agent['unique_evidence_count']}\n")
    lines.append(
        "- Maximum consecutive no-progress rounds: "
        f"{agent['max_consecutive_no_progress']}\n"
    )
    lines.append(
        "- Coverage statuses: "
        f"{_format_failure_taxonomy(agent['coverage']['statuses'])}\n"
    )
    lines.append(f"- Coverage proofs: {agent['coverage']['proofs']}\n")
    lines.append(
        "- Test-generation attempts: "
        f"{agent['coverage']['test_generation_attempts']}\n"
    )


def _append_agent_v2_results(lines: list[str], results: list[dict[str, Any]]) -> None:
    lines.append("\n## Agent V2 Results\n\n")
    lines.append(
        "| Sample ID | Evaluation Mode | Run ID | Final Phase | Waiting | Turns | Tokens | Error |\n"
    )
    lines.append(
        "|-----------|-----------------|--------|-------------|---------|-------|--------|-------|\n"
    )
    for result in results:
        waiting = "yes" if result.get("waiting_for_user") else "no"
        error = result.get("error") or ""
        error = _markdown_table_cell(error)
        lines.append(
            f"| `{_safe_text(result.get('id', ''), 60)}` "
            f"| {_evaluation_mode(result)} "
            f"| `{result.get('run_id', '')}` "
            f"| {result.get('final_phase', '')} "
            f"| {waiting} "
            f"| {result.get('turns_taken', 0)} "
            f"| {result.get('token_used', 0)} "
            f"| {error} |\n"
        )


def _append_coverage_proofs(
    lines: list[str], results: list[dict[str, Any]]
) -> None:
    lines.append("\n## Differential Coverage Proofs\n\n")
    for result in results:
        sample_id = _safe_text(result.get("id"), 300)
        status = _safe_text(result.get("coverage_status"), 40) or "pending"
        lines.append(f"### {sample_id}\n\n")
        lines.append(f"- Status: {status}\n")
        summary = _safe_text(result.get("attempt_outcome_summary"), 200)
        if summary:
            lines.append(f"- Attempt outcome summary: {summary}\n")
        command = _safe_text(result.get("coverage_test_command"), 2_000)
        if command:
            lines.append(f"- Test command: `{command}`\n")
        proof = _safe_coverage_proof(result.get("coverage_proof"))
        if proof is None:
            reason = _safe_text(result.get("coverage_failure_reason"), 100)
            lines.append(f"- Proof: none{f' ({reason})' if reason else ''}\n")
            continue
        lines.append(
            "- Test files: "
            + ", ".join(f"`{_safe_text(path, 500)}`" for path in proof["test_files"])
            + "\n"
        )
        for label, runs in (("Fixed", proof["fixed_runs"]), ("Base", proof["base_runs"])):
            for index, run in enumerate(runs, start=1):
                ids = ", ".join(run["failing_test_ids"]) or "none"
                fingerprint = run["assertion_fingerprint"] or "none"
                lines.append(
                    f"- {label} run {index}: outcome={run['outcome']}; "
                    f"failing_test_ids={ids}; fingerprint={fingerprint}\n"
                )


def _format_failure_taxonomy(taxonomy: dict[str, int]) -> str:
    return "; ".join(
        f"{category}: {count}" for category, count in taxonomy.items()
    ) or "none"


def _append_replay_diagnostics(
    lines: list[str], results: list[dict[str, Any]]
) -> None:
    lines.append("\n## Replay Diagnostics\n\n")
    for result in results:
        run_id = _safe_text(result.get("run_id", ""), 300)
        lines.append(f"### {_safe_text(result.get('id', ''), 300)} (`{run_id}`)\n\n")
        replay = result.get("replay")
        if not replay:
            lines.append(f"- Replay unavailable: {result.get('replay_error') or 'missing'}\n")
            continue

        lines.append(f"- Final phase: {_safe_text(replay.get('current_phase', ''), 100)}\n")
        latest_frame = _latest_decision_frame(replay)
        if latest_frame is None:
            lines.append("- Latest frame: none\n")
        else:
            stage = _safe_text(latest_frame.get("stage", ""), 100)
            frame_id = _safe_text(latest_frame.get("frame_id", ""), 300)
            lines.append(f"- Latest frame: {stage} `{frame_id}`\n")
            if latest_frame.get("selected_hypothesis_id"):
                lines.append(
                    "- Selected hypothesis: "
                    f"{_safe_text(latest_frame['selected_hypothesis_id'], 300)}\n"
                )
            selected = latest_frame.get("selected_hypothesis") or {}
            if selected.get("claim"):
                lines.append(f"- Hypothesis claim: {_safe_text(selected['claim'], 1_000)}\n")
            if latest_frame.get("recommended_action"):
                lines.append(
                    "- Recommended action: "
                    f"{_safe_text(latest_frame['recommended_action'], 100)}\n"
                )
            route = latest_frame.get("route") or {}
            if route.get("route"):
                lines.append(f"- Actual route: {_safe_text(route['route'], 100)}\n")
            for warning in latest_frame.get("warnings", []):
                lines.append(
                    "- Warning: "
                    f"expected {_safe_text(warning.get('expected_phase', ''), 100)} "
                    f"but actual {_safe_text(warning.get('actual_phase', ''), 100)}\n"
                )
            for check in latest_frame.get("next_checks", []):
                lines.append(f"- Next check: {_safe_text(check, 500)}\n")
        for summary in _diagnostic_summary(replay):
            lines.append(f"- Diagnostic summary: {_safe_text(summary, 500)}\n")
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
    text = _safe_text(value)
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
        print(
            "agent_v2 all modes combined: "
            f"{agent_v2['agent_successes']}/{agent_v2['samples']} agent_success "
            f"({agent_v2['agent_success_rate']:.1%})"
        )
        if agent_v2["official_scored_samples"]:
            print(
                "official resolved: "
                f"{agent_v2['official_resolved']}/"
                f"{agent_v2['official_scored_samples']} "
                f"({agent_v2['official_resolved_rate']:.1%})"
            )
        else:
            print("official resolved: not scored")
        print(f"agent_v2 wait:{agent_v2['waiting_for_user']}")
        for mode, mode_metrics in agent_v2["by_evaluation_mode"].items():
            samples = mode_metrics["samples"]
            if not samples:
                continue
            models = ", ".join(mode_metrics["models"]) or "none"
            commits = ", ".join(mode_metrics["commits"]) or "none"
            taxonomy = _format_failure_taxonomy(
                mode_metrics["failure_taxonomy"]
            )
            print(
                f"agent_v2 {mode}: samples={samples} | models={models} | "
                f"commits={commits} | "
                f"agent_success={mode_metrics['agent_successes']}/{samples} "
                f"({mode_metrics['agent_success_rate']:.1%}) | "
                f"decisive taxonomy={taxonomy}"
            )
    print(f"{'='*60}")


def main(argv: list[str] | None = None) -> None:
    """Load results, compute metrics, print summary, write markdown."""
    parser = argparse.ArgumentParser(
        prog="python eval/report.py",
        description="Generate a safe RepoPilot evaluation summary.",
    )
    parser.add_argument("--results-file", default=str(RESULTS_PATH))
    parser.add_argument("--summary-file", default=str(SUMMARY_PATH))
    args = parser.parse_args(argv)
    results_path = Path(args.results_file)
    summary_path = Path(args.summary_file)
    try:
        results = load_results(results_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run `python eval/harness.py` first to generate eval_results.json")
        return

    metrics = compute_metrics(results)

    print_summary(metrics)

    md = generate_markdown(results, metrics)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown report saved to {summary_path}")


if __name__ == "__main__":
    main()
