"""Agent-v2 eval runner with saved-run replay diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

agent_v2 = importlib.import_module("src.new_agent").agent_v2
replay_run = importlib.import_module("src.run_store").replay_run

SAMPLES_PATH = REPO_ROOT / "data" / "samples" / "issues_fixes.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "eval_results.json"
MAX_SAMPLES = 5


def load_samples(n: int = MAX_SAMPLES) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with SAMPLES_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


async def evaluate_agent_v2_sample(
    sample: dict[str, Any],
    idx: int,
    max_retries: int = 3,
    token_budget: int = 50000,
) -> dict[str, Any]:
    issue = sample["issue"]
    repo = sample["repo"]
    patch = sample.get("patch", {})
    signals = sample.get("signals", {})
    issue_url = issue["url"]

    payload = await agent_v2(
        issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        save_final_run=True,
    )
    run_id = payload.get("run_id") or payload.get("trace_id") or ""

    replay: dict[str, Any] | None = None
    replay_error: str | None = None
    if run_id:
        try:
            replay = replay_run(run_id)
        except Exception as exc:  # replay should not hide the eval result
            replay_error = f"{type(exc).__name__}: {exc}"

    return {
        "id": sample["id"],
        "mode": "agent_v2",
        "repo": f"{repo['owner']}/{repo['name']}",
        "issue_url": issue_url,
        "issue_title": issue["title"],
        "actual_files": [file["path"] for file in patch.get("files", [])],
        "has_tests_changed": signals.get("has_tests_changed", False),
        "success": payload.get("success", False),
        "waiting_for_user": payload.get("waiting_for_user", False),
        "final_phase": payload.get("final_phase", ""),
        "run_id": run_id,
        "trace_id": payload.get("trace_id", ""),
        "turns_taken": payload.get("turns_taken", 0),
        "token_used": payload.get("token_used", 0),
        "error": payload.get("error"),
        "agent_payload": payload,
        "replay": replay,
        "replay_error": replay_error,
    }


async def run_agent_v2_eval(
    n_samples: int = MAX_SAMPLES,
    max_retries: int = 3,
    token_budget: int = 50000,
    results_path: Path | str = RESULTS_PATH,
) -> list[dict[str, Any]]:
    samples = load_samples(n_samples)
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        print(f"\n{'='*60}", flush=True)
        print(f"Agent v2 sample {i + 1}/{len(samples)}: {sample['id']}", flush=True)
        print(f"{'='*60}", flush=True)
        results.append(
            await evaluate_agent_v2_sample(
                sample,
                i,
                max_retries=max_retries,
                token_budget=token_budget,
            )
        )

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAgent v2 eval results saved to {path}", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python eval/agent_v2_harness.py",
        description="Run RepoPilot's state-graph agent on eval samples.",
    )
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=50000)
    args = parser.parse_args(argv)

    asyncio.run(
        run_agent_v2_eval(
            n_samples=args.samples,
            max_retries=args.max_retries,
            token_budget=args.token_budget,
        )
    )


if __name__ == "__main__":
    main()
