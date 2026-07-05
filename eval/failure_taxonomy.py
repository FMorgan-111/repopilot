"""Failure taxonomy for RepoPilot evals.

Turns a run's per-attempt error logs into a fine-grained failure classification —
finer than the raw ``failure_kind`` — so we can answer "WHAT is the agent
actually failing on" and compare that distribution across runs (e.g. model A vs
model B). This is the observability layer: you cannot optimize what you cannot
measure, and a bare ``FAILED`` tells you nothing about where to invest.

Categories (a failed attempt maps to exactly one):
  - resolved            : the run succeeded (terminal, not a failure)
  - wrong_file_path     : patch targeted a file absent from the repo
  - invalid_diff        : model emitted a unified diff the executor rejects
  - search_not_found    : search block does not exist (anchor hallucination)
  - test_failed         : patch applied, pytest ran and failed (fix logic wrong)
  - infra               : clone / network / timeout — not the agent's fault
  - budget              : token/round budget exhausted
  - other               : unclassified (bucket to keep taxonomy honest)

Usage:
  python -m eval.failure_taxonomy                      # classify latest results
  python -m eval.failure_taxonomy a.json b.json        # compare two runs
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "eval" / "eval_results.json"

CATEGORIES = [
    "resolved",
    "wrong_file_path",
    "invalid_diff",
    "empty_patch",
    "search_not_found",
    "test_failed",
    "infra",
    "budget",
    "other",
]


def classify_attempt(failure_kind: str, error_log: str) -> str:
    """Map one fix attempt to a fine-grained failure category."""
    kind = (failure_kind or "").strip()
    log = (error_log or "")
    low = log.lower()

    if kind == "infra_error" or "infrastructure error" in low:
        return "infra"
    if "readtimeout" in low or "timed out" in low or "remoteprotocolerror" in low:
        return "infra"
    if "token budget" in low or "budget exceeded" in low:
        return "budget"
    if kind == "patch_apply_failed" or kind == "":
        if "target file was not found" in low or "no such file" in low:
            return "wrong_file_path"
        if "search block was not found" in low or "was not found in" in low:
            return "search_not_found"
        if "no valid patches in input" in low:
            # git apply on an EMPTY patch — the model produced no diff (its
            # search/replace was cleared by a gate, or nothing was emitted).
            # This is NOT the model emitting a bad unified diff; keeping it
            # separate stops it inflating invalid_diff (which it did before,
            # making search-hallucination look like a diff-format problem).
            return "empty_patch"
        if "corrupt patch" in low or "diff --git" in low or "@@ " in low:
            return "invalid_diff"
        if "preflight check failed" in low:
            # Preflight rejected a real diff (has hunks but they don't apply).
            return "invalid_diff"
        if kind == "patch_apply_failed":
            return "search_not_found"  # generic apply failure ≈ anchor miss
    if kind == "test_failed":
        return "test_failed"
    if kind == "execution_error":
        return "infra"
    return "other"


def classify_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Classify a whole sample: its decisive (terminal) failure and per-attempt
    categories. The decisive category is the LAST attempt's — that's what the
    run ultimately died on."""
    payload = sample.get("agent_payload") or {}
    attempts = payload.get("fix_attempts") or []
    if sample.get("success"):
        return {"id": sample.get("id"), "decisive": "resolved", "attempts": ["resolved"]}

    per_attempt = [
        classify_attempt(a.get("failure_kind", ""), a.get("error_log", ""))
        for a in attempts
    ]
    if per_attempt:
        decisive = per_attempt[-1]
    else:
        # No attempt recorded — the gate cleared the patch in PLAN before it
        # reached EXECUTE. Classify from the top-level failure_reason.
        err = (sample.get("error") or "").lower()
        if "search blocks that do not exist" in err or "search block" in err:
            decisive = "search_not_found"  # search-content hallucination
        elif "re-emitting patches that already failed" in err:
            decisive = "search_not_found"  # dead-patch (repeated bad anchor)
        elif "no relevant files" in err or "locate" in err or "context collection" in err:
            decisive = "other"  # pre-patch (localization) failure
        elif "readtimeout" in err or "timed out" in err or "infrastructure" in err:
            decisive = "infra"
        elif "generate fix plan" in err:
            decisive = "infra"
        else:
            decisive = "other"
    return {"id": sample.get("id"), "decisive": decisive, "attempts": per_attempt}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate decisive-failure distribution across a run's samples."""
    classified = [classify_sample(s) for s in results]
    decisive = Counter(c["decisive"] for c in classified)
    attempt_totals = Counter()
    for c in classified:
        attempt_totals.update(c["attempts"])
    n = len(results)
    resolved = decisive.get("resolved", 0)
    return {
        "n_samples": n,
        "resolved": resolved,
        "resolve_rate": (resolved / n) if n else 0.0,
        "decisive": {cat: decisive.get(cat, 0) for cat in CATEGORIES if decisive.get(cat)},
        "attempt_totals": {
            cat: attempt_totals.get(cat, 0) for cat in CATEGORIES if attempt_totals.get(cat)
        },
        "samples": classified,
    }


def format_summary(summary: dict[str, Any], label: str = "") -> str:
    lines = []
    head = f"Failure taxonomy{' — ' + label if label else ''}"
    lines.append(head)
    lines.append("=" * len(head))
    lines.append(
        f"resolved: {summary['resolved']}/{summary['n_samples']} "
        f"({summary['resolve_rate']:.0%})"
    )
    lines.append("\ndecisive failure (what each run finally died on):")
    for cat in CATEGORIES:
        c = summary["decisive"].get(cat)
        if c:
            bar = "█" * c
            lines.append(f"  {cat:18s} {c:2d}  {bar}")
    lines.append("\nper-attempt totals (every attempt across all samples):")
    for cat in CATEGORIES:
        c = summary["attempt_totals"].get(cat)
        if c:
            lines.append(f"  {cat:18s} {c}")
    return "\n".join(lines)


def format_diff(a: dict[str, Any], b: dict[str, Any], la: str, lb: str) -> str:
    """Side-by-side decisive-failure comparison of two runs."""
    lines = [f"Run comparison: {la}  vs  {lb}", "=" * 40]
    lines.append(
        f"resolve rate: {a['resolve_rate']:.0%} ({a['resolved']}/{a['n_samples']})"
        f"  vs  {b['resolve_rate']:.0%} ({b['resolved']}/{b['n_samples']})"
    )
    lines.append(f"\n{'category':18s} {la:>8s} {lb:>8s}  Δ")
    for cat in CATEGORIES:
        ca = a["decisive"].get(cat, 0)
        cb = b["decisive"].get(cat, 0)
        if ca or cb:
            lines.append(f"{cat:18s} {ca:8d} {cb:8d}  {cb - ca:+d}")
    return "\n".join(lines)


def _load(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) >= 2:
        a, b = _load(args[0]), _load(args[1])
        print(format_diff(summarize(a), summarize(b), Path(args[0]).stem, Path(args[1]).stem))
    else:
        path = args[0] if args else DEFAULT_RESULTS
        print(format_summary(summarize(_load(path)), Path(path).stem))


if __name__ == "__main__":
    main()
