import pytest
from pydantic import ValidationError

from src.schemas import PlanDecision, ReflectDecision


def test_plan_decision_requires_plan_frame():
    decision = PlanDecision.model_validate(
        {
            "plan": "Patch auth submit handling.",
            "patch": "diff --git a/src/auth.py b/src/auth.py",
            "files": ["src/auth.py"],
            "test_command": "pytest tests/test_auth.py -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Patch auth submit handling.",
                "recommended_action": "execute",
                "confidence": 0.84,
                "risk": "medium",
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "Submit path mishandles missing input.",
                        "evidence": ["Issue reports a crash after submit."],
                        "score": 0.84,
                    }
                ],
                "selected_hypothesis_id": "H1",
                "next_checks": ["Run auth regression tests."],
            },
        }
    )

    assert decision.decision_frame.stage == "plan"
    assert decision.decision_frame.recommended_action == "execute"


def test_plan_decision_rejects_reflect_frame():
    with pytest.raises(ValidationError):
        PlanDecision.model_validate(
            {
                "plan": "Patch auth submit handling.",
                "patch": "diff --git a/src/auth.py b/src/auth.py",
                "files": ["src/auth.py"],
                "test_command": "pytest tests/test_auth.py -q",
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "Wrong stage.",
                    "recommended_action": "plan",
                },
            }
        )


def test_reflect_decision_requires_reflect_frame():
    decision = ReflectDecision.model_validate(
        {
            "root_cause": "The patch changed the wrong branch.",
            "what_went_wrong": "It ignored the failing None case.",
            "suggested_fix_approach": "Patch the None guard before submit.",
            "files_that_also_need_changes": ["src/auth.py"],
            "decision_frame": {
                "stage": "reflect",
                "summary": "The patch changed the wrong branch.",
                "recommended_action": "plan",
                "confidence": 0.9,
                "risk": "low",
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "Previous patch targeted the wrong condition.",
                        "evidence": ["Test output still fails on None input."],
                        "score": 0.9,
                    }
                ],
                "selected_hypothesis_id": "H1",
                "next_checks": ["Re-run the failing auth test."],
            },
        }
    )

    assert decision.decision_frame.stage == "reflect"
    assert decision.decision_frame.recommended_action == "plan"


def test_reflect_decision_rejects_plan_frame():
    with pytest.raises(ValidationError):
        ReflectDecision.model_validate(
            {
                "root_cause": "The patch changed the wrong branch.",
                "what_went_wrong": "It ignored the failing None case.",
                "suggested_fix_approach": "Patch the None guard before submit.",
                "files_that_also_need_changes": ["src/auth.py"],
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Wrong stage.",
                    "recommended_action": "execute",
                },
            }
        )
