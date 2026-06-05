"""Pydantic schemas for structured LLM outputs — enforce contracts, not hope."""
from pydantic import BaseModel, Field
from typing import Optional


class Classification(BaseModel):
    type: str = Field(..., pattern="^(bug|feature|docs|test|security|unknown)$")
    severity: str = Field(..., pattern="^(low|medium|high|unknown)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""


class RankedFile(BaseModel):
    path: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class FileRanking(BaseModel):
    files: list[RankedFile]


class FixPlan(BaseModel):
    fix_plan: str
    risk_level: str = Field(..., pattern="^(low|medium|high|unknown)$")
    test_suggestions: list[str] = []


class AgentAction(BaseModel):
    tool: Optional[str] = None
    args: dict = Field(default_factory=dict)


class AgentResult(BaseModel):
    done: bool = True
    summary: str = ""
    files: list[str] = Field(default_factory=list)
    fix_plan: str = ""
