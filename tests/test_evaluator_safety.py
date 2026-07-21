from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.evaluator_safety import sanitize_evaluator_text
from src.state import AgentState, FixAttempt, VerifiedEdit


@pytest.mark.parametrize(
    "payload",
    [
        'safe prefix\n{\n  "gold_patch": {\n    "value": "pretty-json-sentinel"\n  }\n}',
        "safe prefix\ngold_patch: |\n  yaml-literal-sentinel\nnext: leaked",
        "safe prefix\nFAIL_TO_PASS:\n  - yaml-list-sentinel\nafter: leaked",
        "safe prefix\n```evaluator\nmarkdown-fence-sentinel\n```\nafter leaked",
    ],
)
def test_sanitizer_truncates_everything_from_first_evaluator_marker_line(payload):
    sanitized = sanitize_evaluator_text(payload)
    assert sanitized.startswith("safe prefix")
    assert "sentinel" not in sanitized
    assert "leaked" not in sanitized
    assert "gold_patch" not in sanitized.casefold()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_path", "tests/test_gold_patch.py"),
        ("node_target", "FAIL_TO_PASS.target"),
        ("search", "value = 1\n# evaluator marker"),
        ("replace", "value = 2\n# oracle marker"),
        ("intent", "Apply the grading payload"),
    ],
)
def test_verified_edit_rejects_evaluator_marker_in_every_model_field(field, value):
    values = {
        "file_path": "tests/test_safe.py",
        "node_target": None,
        "search": "value = 1",
        "replace": "value = 2",
        "intent": "Apply the bounded edit",
    }
    values[field] = value
    if field == "node_target":
        values["search"] = ""
    with pytest.raises(ValidationError, match="evaluator"):
        VerifiedEdit(**values)


def test_agent_state_rejects_contaminated_patch_and_fix_attempt_payloads():
    sentinel = "legacy-patch-sentinel-7712"
    with pytest.raises(ValidationError, match="evaluator"):
        AgentState(
            issue_url="https://github.com/acme/widget/issues/7",
            patch_content=f"gold_patch={sentinel}",
            fix_attempts=[
                FixAttempt(
                    patch_content=f"FAIL_TO_PASS: {sentinel}",
                    success=True,
                )
            ],
        )


def test_sanitized_payload_contains_no_marker_tail_as_json():
    sentinel = "nested-json-sentinel-1148"
    rendered = json.dumps(
        {
            "safe": sanitize_evaluator_text(
                f"safe\ninstance_id:\n  nested:\n    value: {sentinel}"
            )
        }
    )
    assert sentinel not in rendered
