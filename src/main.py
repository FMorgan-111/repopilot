"""RepoPilot — AI Agent that turns GitHub issues into fix plans."""
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.agent import analyze_issue
from src.agent_loop import agent_analyze
from src.new_agent import agent_v2, intelligent_analyze_issue

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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
