"""Agent orchestrator — runs the full analysis pipeline."""
import re
import httpx
from .tools import read_issue, search_code
from .llm import classify_issue, rank_files, generate_fix_plan
from .tracer import Tracer


def parse_issue_url(url: str) -> tuple[str, str, int]:
    """Extract (owner, repo, issue_number) from GitHub issue URL."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
    if not m:
        raise ValueError(f"Invalid GitHub issue URL: {url}")
    return m.group(1), m.group(2), int(m.group(3))


async def analyze_issue(issue_url: str) -> dict:
    """Full pipeline: read → classify → search → rank → plan."""
    t = Tracer()

    # Step 1: Parse URL
    try:
        owner, repo, num = parse_issue_url(issue_url)
    except ValueError as e:
        t.log("parse_url", {"url": issue_url}, {}, error=str(e))
        return {"error": str(e), "trace_id": t.trace_id}

    t.log("parse_url", {"url": issue_url}, {"owner": owner, "repo": repo, "number": num})

    # Step 2: Fetch issue
    try:
        issue = await read_issue(owner, repo, num)
    except httpx.HTTPError as e:
        t.log("read_issue", {"owner": owner, "repo": repo, "number": num},
              {}, error=f"HTTP {e}")
        return {"error": f"Failed to read issue: {e}", "trace_id": t.trace_id}

    t.log("read_issue", {"repo": f"{owner}/{repo}", "number": num},
          {"title": issue["title"], "labels": issue["labels"]})

    # Step 3: Classify
    try:
        classification = await classify_issue(issue["title"], issue["body"])
    except Exception as e:
        t.log("classify", {"title": issue["title"]}, {}, error=str(e))
        classification = {"type": "unknown", "severity": "unknown", "confidence": 0.0}

    t.log("classify", {"title": issue["title"]}, classification)

    # Step 4: Search code
    try:
        # Build search query from title + body start (better than title alone)
        raw = f"{issue['title']} {issue['body'][:200]}"
        # Keep only alphanumeric, spaces, dots, underscores
        query = ' '.join(w for w in raw.replace('/', ' ').split() if len(w) > 1)[:200]
        files = await search_code(query, owner, repo)
    except Exception as e:
        t.log("search_code", {"query": query, "repo": f"{owner}/{repo}"},
              {}, error=str(e))
        files = []

    t.log("search_code", {"query": query}, {"count": len(files)})

    # Step 5: Rank files
    ranked = []
    if files:
        try:
            ranked = await rank_files(issue["title"], issue["body"], files)
        except Exception as e:
            t.log("rank_files", {"files": len(files)}, {}, error=str(e))
            ranked = files  # fallback: use unranked list

    t.log("rank_files", {"input_count": len(files)},
          {"output_count": len(ranked)})

    # Step 6: Generate fix plan
    try:
        plan = await generate_fix_plan(
            issue["title"], issue["body"], classification, ranked
        )
    except Exception as e:
        t.log("fix_plan", {}, {}, error=str(e))
        plan = {"fix_plan": "Could not generate plan.", "risk_level": "unknown", "test_suggestions": []}

    t.log("fix_plan", {}, {"risk_level": plan.get("risk_level")})

    return {
        "trace_id": t.trace_id,
        "issue": {
            "title": issue["title"],
            "state": issue["state"],
            "labels": issue["labels"],
        },
        "classification": classification,
        "files": ranked,
        "fix_plan": plan.get("fix_plan", ""),
        "risk_level": plan.get("risk_level", "unknown"),
        "test_suggestions": plan.get("test_suggestions", []),
    }
