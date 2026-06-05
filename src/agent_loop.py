"""Real agent — an LLM-driven tool-calling loop.

Unlike `agent.analyze_issue` (a fixed linear pipeline), here the LLM decides
which tool to call at each step and loops until it returns a final answer or
hits `max_turns`. The linear pipeline in `agent.py` is kept as a baseline.

`llm.llm_call` takes a system prompt and a single user prompt, so each turn we
pass `SYSTEM_PROMPT` as the system message and a running transcript (the task
plus prior tool calls and results) as the user message.
"""
import json

from .tools import read_issue, search_code, read_file
from .llm import llm_call
from .tracer import Tracer
from .agent import parse_issue_url

TOOLS = [
    {"name": "read_issue", "desc": "Read the GitHub issue title and body. args: {}"},
    {"name": "search_code", "desc": 'Search code in the GitHub repo. args: {"query": <string>}'},
    {"name": "read_file", "desc": 'Read a file\'s contents from the GitHub repo. args: {"path": <string>}'},
]

_TOOL_LINES = "\n".join(f'- {t["name"]}: {t["desc"]}' for t in TOOLS)

SYSTEM_PROMPT = (
    "You are a code analysis agent. You analyze a GitHub issue using tools.\n"
    "Available tools:\n"
    f"{_TOOL_LINES}\n\n"
    "At each step respond with ONLY a JSON object — no markdown, no prose.\n"
    'To call a tool: {"tool": "<name>", "args": {...}}\n'
    'When finished: {"done": true, "summary": "<analysis>", '
    '"files": [<relevant paths>], "fix_plan": "<plan>"}\n'
    "Gather context with tools before answering. Stop once you have enough."
)


async def execute_tool(tool_name: str, tool_args: dict, owner: str, repo: str, issue_number: int):
    """Dispatch a tool call to the GitHub helpers. Returns a JSON-serializable result."""
    if tool_name == "read_issue":
        return await read_issue(owner, repo, issue_number)
    if tool_name == "search_code":
        return await search_code(tool_args.get("query", ""), owner, repo)
    if tool_name == "read_file":
        return await read_file(owner, repo, tool_args.get("path", ""))
    raise ValueError(f"Unknown tool: {tool_name}")


async def agent_analyze(issue_url: str, max_turns: int = 10) -> dict:
    """Run the agent loop: the LLM picks tools until it is done or max_turns is hit."""
    t = Tracer()

    try:
        owner, repo, issue_number = parse_issue_url(issue_url)
    except ValueError as e:
        t.log("parse_url", {"url": issue_url}, {}, error=str(e))
        return {"error": str(e), "trace_id": t.trace_id}

    transcript = f"Analyze this GitHub issue: {issue_url}\nDecide your first step."

    for turn in range(max_turns):
        try:
            response = await llm_call(SYSTEM_PROMPT, transcript)
        except Exception as e:
            t.log("llm_call", {"turn": turn}, {}, error=str(e))
            return {"done": True, "error": f"LLM call failed: {e}",
                    "trace_id": t.trace_id, "turns": turn}

        if response.get("done"):
            t.log("done", {"turn": turn}, {"keys": list(response.keys())})
            return {**response, "trace_id": t.trace_id, "turns": turn + 1}

        tool_name = response.get("tool")
        tool_args = response.get("args") or {}

        if not tool_name:
            t.log("decide", {"turn": turn}, {}, error="response had no 'tool' or 'done'")
            transcript += "\n\nYour last response had no 'tool' or 'done'. Respond with valid JSON."
            continue

        try:
            tool_result = await execute_tool(tool_name, tool_args, owner, repo, issue_number)
            error = None
        except Exception as e:
            tool_result = {"error": str(e)}
            error = str(e)

        logged = tool_result if isinstance(tool_result, dict) else {"result": tool_result}
        t.log(tool_name, tool_args, logged, error=error)

        transcript += (
            f"\n\nYou called: {json.dumps(response)}\n"
            f"Tool result: {json.dumps(tool_result)}\n"
            "Decide your next step."
        )

    t.log("max_turns", {"max_turns": max_turns}, {})
    return {"done": True, "error": "Max turns reached",
            "trace_id": t.trace_id, "turns": max_turns}
