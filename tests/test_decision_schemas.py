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


def test_plan_decision_accepts_search_replace_patch_edits():
    decision = PlanDecision.model_validate(
        {
            "plan": "Patch auth submit handling.",
            "patch_edits": [
                {
                    "file": "src/auth.py",
                    "search": "if retry_count > max_retries:\n    return FAILURE\n",
                    "replace": "if retry_count >= max_retries:\n    return FAILURE\n",
                }
            ],
            "files": ["src/auth.py"],
            "test_command": "pytest tests/test_auth.py -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Patch auth submit handling.",
                "recommended_action": "execute",
                "confidence": 0.84,
                "risk": "medium",
            },
        }
    )

    assert decision.patch == ""
    assert decision.patch_edits[0].file_path == "src/auth.py"
    assert decision.patch_edits[0].search.startswith("if retry_count >")
    assert decision.patch_edits[0].replace.startswith("if retry_count >=")
    assert decision.patch_edits[0].replace_all is False


def test_plan_decision_normalizes_string_evidence_to_lists():
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
                "evidence": "Issue reports a crash after submit.",
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "Submit path mishandles missing input.",
                        "evidence": "Trace points at the submit handler.",
                        "score": 0.84,
                    }
                ],
                "selected_hypothesis_id": "H1",
                "next_checks": ["Run auth regression tests."],
            },
        }
    )

    assert decision.decision_frame.evidence == ["Issue reports a crash after submit."]
    assert decision.decision_frame.hypotheses[0].evidence == [
        "Trace points at the submit handler."
    ]


def test_plan_decision_normalizes_hypothesis_score_from_ten_point_scale():
    decision = PlanDecision.model_validate(
        {
            "plan": "Patch auth submit handling.",
            "patch_edits": [
                {
                    "file": "src/auth.py",
                    "search": "old",
                    "replace": "new",
                }
            ],
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
                        "score": 9,
                    }
                ],
            },
        }
    )

    assert decision.decision_frame.hypotheses[0].score == 0.9


def test_plan_decision_coerces_integer_hypothesis_ids_to_strings():
    decision = PlanDecision.model_validate(
        {
            "plan": "Patch auth submit handling.",
            "patch_edits": [
                {
                    "file": "src/auth.py",
                    "search": "old",
                    "replace": "new",
                }
            ],
            "decision_frame": {
                "stage": "plan",
                "summary": "Patch auth submit handling.",
                "recommended_action": "execute",
                "confidence": 0.84,
                "risk": "medium",
                "hypotheses": [
                    {
                        "id": 1,
                        "claim": "Submit path mishandles missing input.",
                        "score": 0.84,
                    }
                ],
                "selected_hypothesis_id": 1,
            },
        }
    )

    assert decision.decision_frame.hypotheses[0].id == "1"
    assert decision.decision_frame.selected_hypothesis_id == "1"


def test_decision_frame_normalizes_none_evidence_and_next_checks_to_empty_lists():
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
                "evidence": None,
                "next_checks": None,
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "Submit path mishandles missing input.",
                        "evidence": None,
                        "score": 0.84,
                    }
                ],
                "selected_hypothesis_id": "H1",
            },
        }
    )

    assert decision.decision_frame.evidence == []
    assert decision.decision_frame.next_checks == []
    assert decision.decision_frame.hypotheses[0].evidence == []


def test_plan_decision_normalizes_scalar_files_to_list():
    decision = PlanDecision.model_validate(
        {
            "plan": "Patch auth submit handling.",
            "patch": "diff --git a/src/auth.py b/src/auth.py",
            "files": "src/auth.py",
            "test_command": "pytest tests/test_auth.py -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Patch auth submit handling.",
                "recommended_action": "execute",
                "confidence": 0.84,
                "risk": "medium",
            },
        }
    )

    assert decision.files == ["src/auth.py"]


def test_reflect_decision_normalizes_scalar_files_to_list():
    decision = ReflectDecision.model_validate(
        {
            "root_cause": "The patch changed the wrong branch.",
            "what_went_wrong": "It ignored the failing None case.",
            "suggested_fix_approach": "Patch the None guard before submit.",
            "files_that_also_need_changes": "src/auth.py",
            "decision_frame": {
                "stage": "reflect",
                "summary": "The patch changed the wrong branch.",
                "recommended_action": "plan",
                "confidence": 0.9,
                "risk": "low",
            },
        }
    )

    assert decision.files_that_also_need_changes == ["src/auth.py"]


def test_plan_and_reflect_decisions_normalize_none_files_to_empty_lists():
    plan = PlanDecision.model_validate(
        {
            "plan": "Patch auth submit handling.",
            "patch": "diff --git a/src/auth.py b/src/auth.py",
            "files": None,
            "test_command": "pytest tests/test_auth.py -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Patch auth submit handling.",
                "recommended_action": "execute",
                "confidence": 0.84,
                "risk": "medium",
            },
        }
    )
    reflect = ReflectDecision.model_validate(
        {
            "root_cause": "The patch changed the wrong branch.",
            "what_went_wrong": "It ignored the failing None case.",
            "suggested_fix_approach": "Patch the None guard before submit.",
            "files_that_also_need_changes": None,
            "decision_frame": {
                "stage": "reflect",
                "summary": "The patch changed the wrong branch.",
                "recommended_action": "plan",
                "confidence": 0.9,
                "risk": "low",
            },
        }
    )

    assert plan.files == []
    assert reflect.files_that_also_need_changes == []


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
