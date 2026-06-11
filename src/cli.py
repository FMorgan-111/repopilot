"""RepoPilot CLI — AI-powered GitHub issue to fix PR."""
import argparse
import asyncio
import json
import sys

from .new_agent import agent_v2, resume_agent_v2
from .run_store import format_replay_markdown, inspect_run, list_runs, replay_run


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "resume":
        result = _run_resume(argv[1:])
    elif argv and argv[0] == "runs":
        result = _run_runs(argv[1:])
    elif argv and argv[0] == "inspect":
        result = _run_inspect(argv[1:])
    elif argv and argv[0] == "replay":
        result = _run_replay(argv[1:])
    else:
        result = _run_issue(argv)

    success = result.get("success", True) if isinstance(result, dict) else True
    sys.exit(0 if success else 1)


def _run_issue(argv: list[str]) -> dict:
    parser = argparse.ArgumentParser(
        prog="repopilot",
        description="AI agent that reads a GitHub Issue, searches code, "
        "generates fix, runs tests, creates PR.",
    )
    parser.add_argument("issue_url", help="GitHub Issue URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze but don't create PR",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts (default: 3)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=50000,
        help="Token budget (default: 50000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args(argv)

    if not args.json:
        print(f"RepoPilot analyzing {args.issue_url}...")

    result = asyncio.run(
        agent_v2(
            args.issue_url,
            max_retries=args.max_retries,
            token_budget=args.token_budget,
        )
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return result


def _run_resume(argv: list[str]) -> dict:
    parser = argparse.ArgumentParser(
        prog="repopilot resume",
        description="Resume a paused RepoPilot v2 run with human input.",
    )
    parser.add_argument("run_id", help="Paused run id")
    parser.add_argument("human_answer", help="Answer to the pending human-input request")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args(argv)

    if not args.json:
        print(f"RepoPilot resuming {args.run_id}...")

    result = asyncio.run(resume_agent_v2(args.run_id, args.human_answer))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return result


def _run_runs(argv: list[str]) -> list[dict]:
    parser = argparse.ArgumentParser(
        prog="repopilot runs",
        description="List saved RepoPilot runs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args(argv)
    result = list_runs()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_runs(result)

    return result


def _run_inspect(argv: list[str]) -> dict:
    parser = argparse.ArgumentParser(
        prog="repopilot inspect",
        description="Inspect a saved RepoPilot run.",
    )
    parser.add_argument("run_id", help="Saved run id")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args(argv)
    result = inspect_run(args.run_id)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_run_summary(result)

    return result


def _run_replay(argv: list[str]) -> dict:
    parser = argparse.ArgumentParser(
        prog="repopilot replay",
        description="Replay the white-box decision trace for a saved run.",
    )
    parser.add_argument("run_id", help="Saved run id")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output Markdown summary",
    )

    args = parser.parse_args(argv)
    result = replay_run(args.run_id)

    if args.markdown:
        print(format_replay_markdown(result))
    elif args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_replay(result)

    return result


def _print_human(result: dict):
    """Human-readable output."""
    print(f"\nPhase: {result.get('final_phase', 'unknown')}")
    print(f"Success: {result.get('success', False)}")
    if result.get("pr_url"):
        print(f"PR: {result['pr_url']}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print(f"Turns: {result.get('turns_taken', 0)}")
    print(f"Token used: {result.get('token_used', 0)}")

    relevant = result.get("relevant_files", [])
    if relevant:
        print(f"\nRelevant files ({len(relevant)}):")
        for f in relevant[:5]:
            print(
                f"  {f.get('path', '?')} "
                f"(score: {f.get('relevance_score', 0):.2f})"
            )

    attempts = result.get("fix_attempts", [])
    if attempts:
        print(f"\nFix attempts: {len(attempts)}")
        for i, a in enumerate(attempts):
            status = "✅" if a.get("success") else "❌"
            print(f"  {status} Attempt {i+1}: {a.get('file_path', 'unknown')[:60]}")


def _print_runs(runs: list[dict]) -> None:
    if not runs:
        print("No saved runs.")
        return

    for run in runs:
        print(
            f"{run.get('run_id', '')} "
            f"{run.get('current_phase', '')} "
            f"{run.get('issue_url', '')}"
        )
        question = run.get("human_input_question")
        if question:
            print(f"  Question: {question}")


def _print_run_summary(run: dict) -> None:
    print(f"Run: {run.get('run_id', '')}")
    print(f"Issue: {run.get('issue_url', '')}")
    print(f"Phase: {run.get('current_phase', '')}")
    print(f"Pending human input: {run.get('pending_human_input', False)}")
    if run.get("human_input_question"):
        print(f"Question: {run['human_input_question']}")
    if run.get("updated_at"):
        print(f"Updated: {run['updated_at']}")


def _print_replay(replay: dict) -> None:
    print(f"Run: {replay.get('run_id', '')}")
    print(f"Issue: {replay.get('issue_url', '')}")
    print(f"Phase: {replay.get('current_phase', '')}")
    pause = replay.get("pause") or {}
    if pause.get("question"):
        print(f"Question: {pause['question']}")
    for item in replay.get("timeline", []):
        if item.get("type") == "decision_frame":
            print(
                f"{item.get('index')}. "
                f"{item.get('stage')} {item.get('frame_id')}: "
                f"{item.get('recommended_action')}"
            )
            if item.get("summary"):
                print(f"   {item['summary']}")
        elif item.get("type") == "route_decision":
            route = item.get("route") or {}
            print(f"{item.get('index')}. route: {route.get('route', '')}")
