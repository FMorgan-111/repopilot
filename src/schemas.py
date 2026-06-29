"""Pydantic schemas for structured LLM outputs — enforce contracts, not hope."""
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .state import DecisionFrame, PatchEdit


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Expected a sequence of strings")
            normalized.append(item)
        return normalized
    raise TypeError("Expected None, a string, or a sequence of strings")


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


class PlanDecision(BaseModel):
    plan: str
    patch: str = ""
    patch_edits: list[PatchEdit] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    test_command: str = ""
    decision_frame: DecisionFrame

    @field_validator("files", mode="before")
    @classmethod
    def _normalize_files(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)

    @model_validator(mode="after")
    def _require_plan_frame(self):
        if self.decision_frame.stage != "plan":
            raise ValueError("PlanDecision.decision_frame.stage must be 'plan'")
        return self


class ReflectDecision(BaseModel):
    root_cause: str
    what_went_wrong: str
    suggested_fix_approach: str
    files_that_also_need_changes: list[str] = Field(default_factory=list)
    decision_frame: DecisionFrame

    @field_validator("files_that_also_need_changes", mode="before")
    @classmethod
    def _normalize_files_that_also_need_changes(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)

    @model_validator(mode="after")
    def _require_reflect_frame(self):
        if self.decision_frame.stage != "reflect":
            raise ValueError("ReflectDecision.decision_frame.stage must be 'reflect'")
        return self
