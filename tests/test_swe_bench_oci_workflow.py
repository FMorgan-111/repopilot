from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/swe-bench-oci-eval.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _named_step(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^      - name: {re.escape(name)}\n(?P<body>.*?)(?=^      - (?:name:|uses:)|^  [a-zA-Z_-]+:|\Z)",
        text,
    )
    assert match is not None, f"missing workflow step: {name}"
    return match.group("body")


def test_workflow_is_manual_only_with_fixed_modes() -> None:
    text = _workflow_text()

    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert re.search(r"(?m)^\s*push:", text) is None
    assert "checkpoint_5" in text
    assert "baseline_50" in text
    assert "baseline_10" not in text
    assert "type: choice" in text


def test_workflow_serializes_eval_runs_without_cancelling() -> None:
    text = _workflow_text()

    assert "group: swe-bench-oci-evaluation" in text
    assert "cancel-in-progress: false" in text


def test_workflow_cache_is_public_dataset_only() -> None:
    text = _workflow_text()
    cache = _named_step(text, "Restore public SWE-bench dataset cache")

    assert "actions/cache@v4" in cache
    assert "swe-bench-verified-main-v1" in cache
    assert "public-hf-cache" in cache
    assert "repopilot-home/eval/datasets" in cache
    for forbidden in (
        "llm_api_key",
        "llm_escalation_api_key",
        "prediction",
        "result.json",
        "target checkout",
        "docker",
    ):
        assert forbidden not in cache.casefold()


def test_workflow_uses_one_instance_per_job_and_two_way_bound() -> None:
    text = _workflow_text()

    assert "max-parallel: 2" in text
    assert "fail-fast: false" in text
    assert "matrix: ${{ fromJSON(needs.prepare.outputs.matrix) }}" in text
    assert "INSTANCE_ID: ${{ matrix.instance_id }}" in text
    assert "load_mode_instance_ids" in text


def test_model_secrets_exist_only_in_generation_step() -> None:
    text = _workflow_text()
    generate = _named_step(text, "Generate patch")
    without_generate = text.replace(generate, "")

    assert "secrets.LLM_API_KEY" in generate
    assert "secrets.LLM_ESCALATION_API_KEY" in generate
    assert "secrets.LLM_API_KEY" not in without_generate
    assert "secrets.LLM_ESCALATION_API_KEY" not in without_generate
    assert "set -x" not in text


def test_score_package_and_upload_are_failure_tolerant_and_sanitized() -> None:
    text = _workflow_text()

    for step_name in (
        "Score prediction without model credentials",
        "Package safe artifact",
        "Upload safe instance artifact",
    ):
        assert "if: always()" in _named_step(text, step_name)
    package = _named_step(text, "Package safe artifact")
    upload = _named_step(text, "Upload safe instance artifact")
    assert '--artifact-dir "$RUNNER_TEMP/upload"' in package
    assert "path: ${{ runner.temp }}/upload" in upload
    assert "runner.temp }}/instance" not in upload


def test_workflow_pins_permissions_and_aggregates_current_commit() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read" in text
    assert "persist-credentials: false" in text
    assert '--expected-commit "${{ github.sha }}"' in text
    assert "python -m eval.oci_aggregate" in text
    assert "retention-days: 30" in text
