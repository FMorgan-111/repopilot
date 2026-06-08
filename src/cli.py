"""RepoPilot CLI — AI-powered GitHub issue to fix PR."""
import argparse
import asyncio
import sys

from .new_agent import agent_v2


def main():
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

    args = parser.parse_args()

    if not args.json:
        print(f"🔍 RepoPilot analyzing {args.issue_url}...")

    result = asyncio.run(
        agent_v2(
            args.issue_url,
            max_retries=args.max_retries,
            token_budget=args.token_budget,
        )
    )

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    sys.exit(0 if result.get("success") else 1)


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
