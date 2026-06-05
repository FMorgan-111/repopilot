"""RepoPilot — AI Agent that turns GitHub issues into fix plans."""
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import analyze_issue
from src.agent_loop import agent_analyze

app = FastAPI(title="RepoPilot")


class AnalyzeRequest(BaseModel):
    issue_url: str


class AgentRequest(BaseModel):
    issue_url: str
    max_turns: int = 10


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    result = await analyze_issue(req.issue_url)
    if "error" in result:
        return {"status": "error", **result}
    return result


@app.post("/agent")
async def agent(req: AgentRequest):
    result = await agent_analyze(req.issue_url, req.max_turns)
    if "error" in result:
        return {"status": "error", **result}
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
