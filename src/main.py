"""RepoPilot — AI Agent that turns GitHub issues into fix plans."""
# ruff: noqa: E402,I001
import json

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError
from src.agent import analyze_issue
from src.agent_loop import agent_analyze
from src.new_agent import agent_v2, intelligent_analyze_issue, resume_agent_v2
from src.run_store import format_replay_markdown, replay_run

app = FastAPI(title="RepoPilot")


class AnalyzeRequest(BaseModel):
    issue_url: str


class AgentRequest(BaseModel):
    issue_url: str
    max_turns: int = 10


class IntelligentAgentRequest(BaseModel):
    issue_url: str
    max_turns: int = 10
    token_budget: int = 100000


class AgentV2Request(BaseModel):
    issue_url: str
    max_retries: int = 3
    token_budget: int = 50000


class AgentV2ResumeRequest(BaseModel):
    run_id: str
    human_answer: str


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """基础线性 pipeline 分析"""
    result = await analyze_issue(req.issue_url)
    if "error" in result:
        status = 400 if "Invalid" in result["error"] else 502
        return JSONResponse({"status": "error", **result}, status_code=status)
    return result


@app.post("/agent")
async def agent(req: AgentRequest):
    """简单 LLM 循环 agent"""
    result = await agent_analyze(req.issue_url, req.max_turns)
    if "error" in result:
        status = 400 if "Invalid" in result["error"] else 502
        return JSONResponse({"status": "error", **result}, status_code=status)
    return result


@app.post("/intelligent-agent")
async def intelligent_agent(req: IntelligentAgentRequest):
    """🚀 新的智能推理 agent - 带状态机和执行反馈循环"""
    result = await intelligent_analyze_issue(
        req.issue_url,
        req.max_turns,
        req.token_budget
    )
    if result.get("error"):
        status = 400 if "Invalid" in result["error"] else 502
        return JSONResponse({"status": "error", **result}, status_code=status)
    return result


@app.post("/agent/v2")
async def agent_v2_endpoint(req: AgentV2Request):
    """State-graph agent with execute/test/replan feedback loop."""
    result = await agent_v2(
        req.issue_url,
        max_retries=req.max_retries,
        token_budget=req.token_budget,
    )
    if result.get("error"):
        status = 400 if "Invalid" in result["error"] else 502
        return JSONResponse({"status": "error", **result}, status_code=status)
    return result


@app.post("/agent/v2/resume")
async def agent_v2_resume_endpoint(req: AgentV2ResumeRequest):
    """Resume a paused state-graph agent run with human input."""
    try:
        result = await resume_agent_v2(req.run_id, req.human_answer)
    except FileNotFoundError:
        return _saved_run_error_response(
            req.run_id,
            f"Saved run {req.run_id} was not found.",
            status_code=404,
        )
    except (json.JSONDecodeError, ValidationError):
        return _saved_run_error_response(
            req.run_id,
            f"Saved run {req.run_id} could not be loaded.",
            status_code=500,
        )

    if result.get("error"):
        status = 400 if _is_client_error(result["error"]) else 502
        return JSONResponse({"status": "error", **result}, status_code=status)
    return result


@app.get("/agent/v2/runs/{run_id}/replay")
async def agent_v2_replay_endpoint(run_id: str, format: str = "json"):
    """Replay the white-box decision trace for a saved run."""
    try:
        replay = replay_run(run_id)
    except FileNotFoundError:
        return _saved_run_error_response(
            run_id,
            f"Saved run {run_id} was not found.",
            status_code=404,
        )
    except (json.JSONDecodeError, ValidationError):
        return _saved_run_error_response(
            run_id,
            f"Saved run {run_id} could not be loaded.",
            status_code=500,
        )

    if format == "markdown":
        return PlainTextResponse(
            format_replay_markdown(replay),
            media_type="text/markdown",
        )
    return replay


@app.get("/health")
async def health():
    return {"status": "ok"}


def _is_client_error(error: str) -> bool:
    return "Invalid" in error or "not waiting for user input" in error


def _saved_run_error_response(run_id: str, error: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "success": False,
            "run_id": run_id,
            "error": error,
        },
        status_code=status_code,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
