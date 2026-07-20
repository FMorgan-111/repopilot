"""Agent-v2 eval runner with saved-run replay diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

agent_v2 = importlib.import_module("src.new_agent").agent_v2
replay_run = importlib.import_module("src.run_store").replay_run
close_llm_client = importlib.import_module("src.http_client").close_llm_client
get_llm_model = importlib.import_module("src.http_client")._get_llm_model
close_store = importlib.import_module("src.memory").close_store
AgentState = importlib.import_module("src.state").AgentState
git_clone = importlib.import_module("src.nodes.execute").git_clone
classify_sample = importlib.import_module("eval.failure_taxonomy").classify_sample
_swe_bench = importlib.import_module("eval.swe_bench")
build_agent_seed = _swe_bench.build_agent_seed
load_verified_samples = _swe_bench.load_verified_samples
write_predictions = _swe_bench.write_predictions

SAMPLES_PATH = REPO_ROOT / "data" / "samples" / "issues_fixes.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "eval_results.json"
MAX_SAMPLES = 5


def _configured_model() -> str:
    return get_llm_model()


def _current_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _fallback_results_path() -> Path:
    configured_home = os.getenv("REPOPILOT_HOME")
    if configured_home:
        return Path(configured_home) / "eval" / "eval_results.json"
    return Path("/tmp") / "repopilot" / "eval_results.json"


def _write_results_with_fallback(
    results: list[dict[str, Any]],
    requested_path: Path | str,
) -> Path:
    path = Path(requested_path)
    contents = json.dumps(results, indent=2, ensure_ascii=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path
    except OSError as exc:
        fallback_path = _fallback_results_path()
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_text(contents, encoding="utf-8")
        print(
            "Warning: failed to write agent v2 eval results to "
            f"{path}: {type(exc).__name__}: {exc}; wrote fallback to {fallback_path}",
            file=sys.stderr,
            flush=True,
        )
        return fallback_path


async def _close_shared_resources() -> None:
    cleanup_steps = [
        ("shared LLM client", close_llm_client),
        ("shared memory store", close_store),
    ]
    for resource_name, close_resource in cleanup_steps:
        try:
            await close_resource()
        except Exception as exc:
            print(
                f"Warning: failed to close {resource_name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def load_samples(
    n: int = MAX_SAMPLES, sample_id: str | None = None,
    samples_path: Path | None = None,
) -> list[dict[str, Any]]:
    # Resolve at call time (not as a default arg) so tests that monkeypatch the
    # module SAMPLES_PATH take effect, while an explicit path still overrides.
    if samples_path is None:
        samples_path = SAMPLES_PATH
    # When a specific sample_id is requested, scan the whole file for it and
    # return just that one (ignores n). Otherwise take the first n lines.
    if sample_id:
        with samples_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("id") == sample_id:
                    return [record]
        raise ValueError(f"sample_id not found in dataset: {sample_id}")

    samples: list[dict[str, Any]] = []
    with samples_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _is_doc_path(path: str) -> bool:
    lower = path.lower()
    return (
        lower.endswith((".rst", ".md", ".txt"))
        or "/docs/" in lower
        or lower.startswith("docs/")
        or "changelog" in lower
        or "/news/" in lower
    )


def _build_gold_seed(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Seed relevant_files from the dataset's known changed files, fetching each
    file's content via the GitHub Contents API (single-file, NOT the flaky code
    search). Removes locate from the critical path so eval measures the patch
    stage, not GitHub search rate-limiting. Returns None if no code file could
    be seeded (fall back to the normal locate path)."""
    from eval.harness import _gh_get, fetch_file_content

    repo = sample["repo"]
    issue = sample["issue"]
    owner, name = repo["owner"], repo["name"]
    meta = _gh_get(f"https://api.github.com/repos/{owner}/{name}")
    ref = (meta or {}).get("default_branch") or "main"

    files: list[dict[str, Any]] = []
    for entry in sample.get("patch", {}).get("files", []):
        path = entry.get("path", "")
        if not path or _is_doc_path(path):
            continue
        content = fetch_file_content(owner, name, path, ref)
        if not content:  # added file with no pre-image, or fetch failed → skip
            continue
        files.append(
            {
                "path": path,
                "content": content,
                "relevance_score": 1.0,
                "reason": "seeded from gold changed files (offline eval)",
            }
        )
    if not files:
        return None
    return {
        "owner": owner,
        "repo": name,
        "issue_number": issue.get("number", 0),
        "issue_title": issue["title"],
        "issue_body": issue["body"],
        "relevant_files": files,
    }


async def _build_eval_seed(
    sample: dict[str, Any], seed_gold_files: bool
) -> dict[str, Any] | None:
    """Build an oracle seed or prepare an exact SWE-bench issue seed."""
    if sample.get("source") != "swe-bench-verified":
        return _build_gold_seed(sample) if seed_gold_files else None
    issue = sample["issue"]
    repo = sample["repo"]
    prep = AgentState(
        issue_url=issue["url"],
        owner=repo["owner"],
        repo=repo["name"],
        issue_number=issue["number"],
        repo_ref=sample["base_commit"],
        skip_commit=True,
    )
    repo_path = await git_clone(prep)
    return build_agent_seed(sample, repo_path)


async def evaluate_agent_v2_sample(
    sample: dict[str, Any],
    idx: int,
    max_retries: int = 3,
    token_budget: int = 50000,
    seed_gold_files: bool = False,
) -> dict[str, Any]:
    issue = sample["issue"]
    repo = sample["repo"]
    patch = sample.get("patch", {})
    signals = sample.get("signals", {})
    issue_url = issue["url"]
    is_swe_bench = sample.get("source") == "swe-bench-verified"

    seed = await _build_eval_seed(sample, seed_gold_files)

    payload = await agent_v2(
        issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        save_final_run=True,
        skip_commit=True,
        seed=seed,
    )
    run_id = payload.get("run_id") or payload.get("trace_id") or ""

    replay: dict[str, Any] | None = None
    replay_error: str | None = None
    if run_id:
        try:
            replay = replay_run(run_id)
        except Exception as exc:  # replay should not hide the eval result
            replay_error = f"{type(exc).__name__}: {exc}"

    result = {
        "id": sample["id"],
        "mode": "agent_v2",
        "evaluation_mode": (
            "oracle_files" if seed_gold_files and not is_swe_bench else "end_to_end"
        ),
        "model": _configured_model(),
        "commit_sha": _current_commit_sha(),
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
    if is_swe_bench:
        result.update(
            {
                "instance_id": sample["instance_id"],
                "base_commit": sample["base_commit"],
                "model_patch": payload.get("model_patch", ""),
            }
        )
        result["failure_class"] = classify_sample(result)["decisive"]
    return result


async def run_agent_v2_eval(
    n_samples: int = MAX_SAMPLES,
    max_retries: int = 3,
    token_budget: int = 50000,
    results_path: Path | str = RESULTS_PATH,
    sample_id: str | None = None,
    seed_gold_files: bool = False,
    samples_path: Path | None = None,
    dataset: str = "custom",
    dataset_seed: int = 17,
    predictions_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    try:
        if predictions_path is not None and dataset != "swe-bench-verified":
            raise ValueError(
                "predictions_path requires dataset='swe-bench-verified'"
            )
        if dataset == "swe-bench-verified":
            samples = load_verified_samples(n_samples, dataset_seed)
        elif dataset == "custom":
            samples = load_samples(
                n_samples,
                sample_id=sample_id,
                samples_path=samples_path,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
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
                    seed_gold_files=seed_gold_files,
                )
            )

        path = _write_results_with_fallback(results, results_path)
        print(f"\nAgent v2 eval results saved to {path}", flush=True)
        if predictions_path is not None:
            prediction_file = write_predictions(results, Path(predictions_path))
            print(
                f"SWE-bench predictions saved to {prediction_file}",
                flush=True,
            )
        return results
    finally:
        await _close_shared_resources()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python eval/agent_v2_harness.py",
        description="Run RepoPilot's state-graph agent on eval samples.",
    )
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=50000)
    parser.add_argument("--seed-gold-files", action="store_true",
                        help="Seed relevant_files from dataset gold changed files")
    parser.add_argument("--samples-file", type=str, default=None,
                        help="Path to custom samples JSONL file")
    parser.add_argument(
        "--dataset",
        choices=("custom", "swe-bench-verified"),
        default="custom",
    )
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--predictions-file", type=str, default=None)
    args = parser.parse_args(argv)

    samples_path = Path(args.samples_file) if args.samples_file else SAMPLES_PATH
    asyncio.run(
        run_agent_v2_eval(
            n_samples=args.samples,
            max_retries=args.max_retries,
            token_budget=args.token_budget,
            seed_gold_files=args.seed_gold_files,
            samples_path=samples_path,
            dataset=args.dataset,
            dataset_seed=args.dataset_seed,
            predictions_path=args.predictions_file,
        )
    )


if __name__ == "__main__":
    main()
