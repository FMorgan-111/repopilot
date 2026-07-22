from __future__ import annotations

import re
from pathlib import Path

import pytest

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


def _mapping_block(text: str, key: str, indent: int) -> str:
    lines = text.splitlines()
    target = f"{' ' * indent}{key}:"
    starts = [
        index
        for index, line in enumerate(lines)
        if line == target or line.startswith(f"{target} ")
    ]
    assert len(starts) == 1, f"expected one {target!r} mapping, found {len(starts)}"

    start = starts[0]
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        end += 1
    return "\n".join(lines[start:end])


def _assert_manual_workflow_contract(text: str) -> None:
    triggers = _mapping_block(text, "on", 0)
    assert re.findall(r"(?m)^  ([a-zA-Z_-]+):", triggers) == ["workflow_dispatch"]

    dispatch = _mapping_block(triggers, "workflow_dispatch", 2)
    inputs = _mapping_block(dispatch, "inputs", 4)
    mode = _mapping_block(inputs, "mode", 6)
    assert "        type: choice" in mode
    options = _mapping_block(mode, "options", 8)
    assert [line.strip() for line in options.splitlines()[1:] if line.strip()] == [
        "- checkpoint_5",
        "- baseline_50",
    ]


def _assert_workflow_concurrency_contract(text: str) -> None:
    concurrency = _mapping_block(text, "concurrency", 0)
    fields = {}
    for line in concurrency.splitlines()[1:]:
        if line.strip() and len(line) - len(line.lstrip()) == 2:
            key, value = line.strip().split(":", 1)
            fields[key] = value.strip()
    assert fields == {
        "group": "swe-bench-oci-evaluation",
        "cancel-in-progress": "false",
        "queue": "max",
    }


def _assert_public_dataset_cache_contract(text: str) -> None:
    cache = _named_step(text, "Restore public SWE-bench dataset cache")
    assert re.findall(r"(?m)^        ([a-zA-Z_-]+):", cache) == ["uses", "with"]
    assert re.findall(r"(?m)^        uses: (.+)$", cache) == ["actions/cache@v4"]
    assert re.findall(r"(?m)^          key: (.+)$", cache) == [
        "swe-bench-verified-main-v1"
    ]

    with_block = _mapping_block(cache, "with", 8)
    assert re.findall(r"(?m)^          ([a-zA-Z_-]+):", with_block) == [
        "path",
        "key",
    ]
    path = _mapping_block(with_block, "path", 10)
    assert path.splitlines()[0] == "          path: |"
    assert [line.strip() for line in path.splitlines()[1:] if line.strip()] == [
        "${{ runner.temp }}/public-hf-cache",
        "${{ runner.temp }}/repopilot-home/eval/datasets",
    ]


def test_workflow_is_manual_only_with_fixed_modes() -> None:
    _assert_manual_workflow_contract(_workflow_text())


def test_manual_workflow_contract_rejects_extra_trigger_or_mode() -> None:
    text = _workflow_text()
    mutations = (
        text.replace(
            "  workflow_dispatch:\n",
            "  schedule:\n    - cron: '0 0 * * *'\n  workflow_dispatch:\n",
            1,
        ),
        text.replace(
            "          - baseline_50\n",
            "          - baseline_50\n          - experimental\n",
            1,
        ),
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_manual_workflow_contract(mutation)


def test_workflow_serializes_eval_runs_without_cancelling() -> None:
    _assert_workflow_concurrency_contract(_workflow_text())


def test_workflow_concurrency_contract_rejects_job_local_lookalike() -> None:
    text = _workflow_text()
    original_root = _mapping_block(text, "concurrency", 0)
    valid_root = original_root
    if "  queue: max" not in valid_root:
        valid_root += "\n  queue: max"
    without_root = text.replace(original_root, "", 1)
    job_local = "\n".join(f"    {line}" for line in valid_root.splitlines())
    mutation = without_root.replace("  instance:\n", f"  instance:\n{job_local}\n", 1)

    with pytest.raises(AssertionError):
        _assert_workflow_concurrency_contract(mutation)


def test_workflow_cache_is_public_dataset_only() -> None:
    _assert_public_dataset_cache_contract(_workflow_text())


def test_public_dataset_cache_contract_rejects_near_matches() -> None:
    text = _workflow_text()
    mutations = (
        text.replace("actions/cache@v4", "actions/cache@v4-extra", 1),
        text.replace(
            "key: swe-bench-verified-main-v1",
            "key: swe-bench-verified-main-v1-extra",
            1,
        ),
        text.replace(
            "            ${{ runner.temp }}/repopilot-home/eval/datasets\n",
            "            ${{ runner.temp }}/repopilot-home/eval/datasets\n"
            "            ${{ runner.temp }}/repopilot-home\n",
            1,
        ),
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_public_dataset_cache_contract(mutation)


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
