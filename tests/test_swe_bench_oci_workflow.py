from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

WORKFLOW = Path(".github/workflows/swe-bench-oci-eval.yml")
README = Path("README.md")
DATASET_CACHE_KEY = (
    "swe-bench-verified-"
    "c104f840cc67f8b6eec6f759ebc8b2693d585d4a-"
    "f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c-v2"
)
LOCK_INSTALL = (
    "python -m pip install --require-hashes -r requirements-eval.lock"
)
PROJECT_INSTALL = (
    "python -m pip install --no-deps --no-build-isolation -e ."
)
ACTION_PINS = {
    "actions/checkout": (
        "11d5960a326750d5838078e36cf38b85af677262",
        "v4",
        3,
    ),
    "actions/setup-python": (
        "a26af69be951a213d495a4c3e4e4022e16d87065",
        "v5",
        3,
    ),
    "actions/cache": (
        "0057852bfaa89a56745cba8c7296529d2fc39830",
        "v4",
        1,
    ),
    "actions/upload-artifact": (
        "ea165f8d65b6e75b540449e92b4886f43607fa02",
        "v4",
        2,
    ),
    "actions/download-artifact": (
        "d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "v4",
        1,
    ),
}


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


def _job_block(text: str, name: str) -> str:
    return _mapping_block(_mapping_block(text, "jobs", 0), name, 2)


def _step_blocks(job: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^      - name: (?P<name>[^\n]+)$", job))
    return [
        (
            match.group("name"),
            job[
                match.start() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(job)
                )
            ],
        )
        for index, match in enumerate(matches)
    ]


def _pip_install_commands(text: str) -> list[str]:
    pattern = re.compile(
        r"(?:^|\s)(?:python\s+-m\s+|uv\s+)?pip(?:3)?\s+install(?:\s|$)"
    )
    return [
        line.strip().removeprefix("run: ").strip()
        for line in text.splitlines()
        if pattern.search(line)
    ]


def _assert_action_pins(text: str) -> None:
    uses_lines = [line for line in text.splitlines() if "uses:" in line]
    parsed: list[tuple[str, str, str]] = []
    for line in uses_lines:
        match = re.fullmatch(
            r"\s+uses: ([a-z0-9_.-]+/[a-z0-9_.-]+)@([0-9a-f]{40}) # (v[0-9]+)",
            line,
        )
        assert match is not None, f"mutable or malformed action reference: {line}"
        parsed.append(match.groups())

    assert Counter(parsed) == Counter(
        {
            (action, commit, tag): count
            for action, (commit, tag, count) in ACTION_PINS.items()
        }
    )


def _assert_immutable_install_contract(text: str) -> None:
    execution_steps = {
        "prepare": "Build tracked instance matrix",
        "instance": "Prepare OCI runtime",
        "aggregate": "Aggregate verified artifacts",
    }
    jobs = _mapping_block(text, "jobs", 0)
    assert re.findall(r"(?m)^  ([a-zA-Z_-]+):", jobs) == list(execution_steps)
    all_install_lines = _pip_install_commands(text)
    assert Counter(all_install_lines) == Counter(
        {LOCK_INSTALL: 3, PROJECT_INSTALL: 3}
    )

    for job_name, first_execution in execution_steps.items():
        steps = _step_blocks(_job_block(text, job_name))
        names = [name for name, _block in steps]
        lock_indexes = [
            index for index, (_name, block) in enumerate(steps) if LOCK_INSTALL in block
        ]
        project_indexes = [
            index
            for index, (_name, block) in enumerate(steps)
            if PROJECT_INSTALL in block
        ]
        assert len(lock_indexes) == 1
        assert len(project_indexes) == 1
        assert lock_indexes[0] < project_indexes[0] < names.index(first_execution)

        install_blocks = [
            block for _name, block in steps if _pip_install_commands(block)
        ]
        assert len(install_blocks) == 2
        assert all("secrets." not in block for block in install_blocks)

    instance_steps = _step_blocks(_job_block(text, "instance"))
    instance_names = [name for name, _block in instance_steps]
    generate_index = instance_names.index("Generate patch")
    assert all(
        index < generate_index
        for index, (_name, block) in enumerate(instance_steps)
        if _pip_install_commands(block)
    )


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
    cache_commit, cache_tag, _count = ACTION_PINS["actions/cache"]
    assert re.findall(r"(?m)^        uses: (.+)$", cache) == [
        f"actions/cache@{cache_commit} # {cache_tag}"
    ]
    assert re.findall(r"(?m)^          key: (.+)$", cache) == [
        DATASET_CACHE_KEY
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


def _assert_runner_temp_initialization_contract(text: str) -> None:
    jobs = _mapping_block(text, "jobs", 0)
    job_blocks = [
        _job_block(text, job_name)
        for job_name in re.findall(
            r"(?m)^  ([A-Za-z_][A-Za-z0-9_-]*):",
            jobs,
        )
    ]
    job_envs = [
        _mapping_block(job, "env", 4)
        for job in job_blocks
        if re.search(r"(?m)^    env:(?:[ \t]|$)", job)
    ]
    assert job_envs
    runner_expression = re.compile(r"\$\{\{\s*runner\s*(?:\.|\[)")
    assert all(not runner_expression.search(job_env) for job_env in job_envs), (
        "runner context is unavailable in job-level env"
    )

    instance = _job_block(text, "instance")
    steps = _step_blocks(instance)
    names = [name for name, _block in steps]
    configure_name = "Configure temporary evaluation paths"
    assert names.count(configure_name) == 1

    configure_index = names.index(configure_name)
    configure = dict(steps)[configure_name]
    commands = re.findall(r"(?m)^          (.+)$", configure)
    fail_closed_guards = [
        ': "${RUNNER_TEMP:?RUNNER_TEMP is required}"',
        ': "${GITHUB_ENV:?GITHUB_ENV is required}"',
    ]
    assert commands[:2] == fail_closed_guards, (
        "runner path initialization must fail closed"
    )
    writes = re.findall(
        r'(?m)^          echo "([^"]+)" >> "\$GITHUB_ENV"$',
        configure,
    )
    assert writes == [
        "REPOPILOT_HOME=$RUNNER_TEMP/repopilot-home",
        "HF_HOME=$RUNNER_TEMP/public-hf-cache",
    ]

    for dependent_name in (
        "Restore public SWE-bench dataset cache",
        "Install locked dependencies",
        "Install RepoPilot without dependency resolution",
    ):
        assert configure_index < names.index(dependent_name)


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


def test_workflow_initializes_runner_paths_after_runner_assignment() -> None:
    _assert_runner_temp_initialization_contract(_workflow_text())


def test_runner_path_contract_rejects_job_level_runner_context() -> None:
    text = _workflow_text()
    mutation = text.replace(
        '      REPOPILOT_ESCALATION_ENABLED: "1"\n',
        '      REPOPILOT_ESCALATION_ENABLED: "1"\n'
        "      FORBIDDEN_PATH: ${{ runner.temp }}/forbidden\n",
        1,
    )

    with pytest.raises(AssertionError, match="job-level env"):
        _assert_runner_temp_initialization_contract(mutation)


def test_runner_path_contract_rejects_inline_job_level_runner_context() -> None:
    text = _workflow_text()
    mutation = text.replace(
        "    runs-on: ubuntu-latest\n",
        '    runs-on: ubuntu-latest\n'
        '    env: {FORBIDDEN_PATH: "${{ runner.temp }}/forbidden"}\n',
        1,
    )

    with pytest.raises(AssertionError, match="job-level env"):
        _assert_runner_temp_initialization_contract(mutation)


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        (
            "    runs-on: ubuntu-latest\n",
            "    runs-on: ubuntu-latest\n"
            "    env: {\n"
            '      FORBIDDEN_PATH: "${{ runner.temp }}/forbidden",\n'
            "    }\n",
        ),
        (
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n',
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n'
            "      FORBIDDEN_PATH: ${{runner.temp}}/forbidden\n",
        ),
        (
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n',
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n'
            "      FORBIDDEN_PATH: ${{ runner['temp'] }}/forbidden\n",
        ),
        (
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n',
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n'
            "      FORBIDDEN_PATH: ${{ runner .temp }}/forbidden\n",
        ),
        (
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n',
            '      REPOPILOT_ESCALATION_ENABLED: "1"\n'
            "      FORBIDDEN_PATH: ${{ runner ['temp'] }}/forbidden\n",
        ),
        (
            "    runs-on: ubuntu-latest\n",
            "    runs-on: ubuntu-latest\n"
            "    env: # job environment\n"
            '      FORBIDDEN_PATH: "${{ runner.temp }}/forbidden"\n',
        ),
        (
            "  prepare:\n    runs-on: ubuntu-latest\n",
            "  prepare2:\n"
            "    runs-on: ubuntu-latest\n"
            '    env: {FORBIDDEN_PATH: "${{ runner.temp }}/forbidden"}\n',
        ),
    ),
    ids=(
        "multiline-flow",
        "compact-dot-access",
        "bracket-access",
        "spaced-dot-access",
        "spaced-bracket-access",
        "comment-bearing-block-header",
        "digit-bearing-job-id",
    ),
)
def test_runner_path_contract_rejects_additional_job_level_runner_forms(
    original: str,
    replacement: str,
) -> None:
    text = _workflow_text()
    mutation = text.replace(original, replacement, 1)
    assert mutation != text

    with pytest.raises(AssertionError, match="job-level env"):
        _assert_runner_temp_initialization_contract(mutation)


def test_public_dataset_cache_contract_rejects_near_matches() -> None:
    text = _workflow_text()
    cache_commit, cache_tag, _count = ACTION_PINS["actions/cache"]
    pinned_cache = f"actions/cache@{cache_commit} # {cache_tag}"
    mutations = (
        text.replace(pinned_cache, f"{pinned_cache}-extra", 1),
        text.replace(
            f"key: {DATASET_CACHE_KEY}",
            f"key: {DATASET_CACHE_KEY}-extra",
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


def test_workflow_actions_use_exact_audited_commits_and_tag_comments() -> None:
    _assert_action_pins(_workflow_text())


def test_action_pin_contract_rejects_mutable_or_unapproved_references() -> None:
    text = _workflow_text()
    checkout_commit, checkout_tag, _count = ACTION_PINS["actions/checkout"]
    checkout = f"actions/checkout@{checkout_commit} # {checkout_tag}"
    mutations = (
        text.replace(checkout, "actions/checkout@v4", 1),
        text.replace(checkout, f"actions/checkout@{'0' * 40} # v4", 1),
        text.replace(checkout, f"other/checkout@{checkout_commit} # v4", 1),
        text.replace(checkout, f"actions/checkout@{checkout_commit} # v5", 1),
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_action_pins(mutation)


def test_every_job_uses_only_hash_locked_then_isolated_project_install() -> None:
    _assert_immutable_install_contract(_workflow_text())


def test_install_contract_rejects_unconstrained_or_secret_bearing_steps() -> None:
    text = _workflow_text()
    mutations = (
        text.replace(LOCK_INSTALL, "python -m pip install -r requirements-eval.lock", 1),
        text.replace(PROJECT_INSTALL, 'python -m pip install -e ".[memory,eval]"', 1),
        text.replace(
            f"        run: {LOCK_INSTALL}",
            f"        env:\n          TOKEN: ${{{{ secrets.LLM_API_KEY }}}}\n"
            f"        run: {LOCK_INSTALL}",
            1,
        ),
        text.replace(
            f"        run: {PROJECT_INSTALL}",
            f"        run: {PROJECT_INSTALL} && python -m pip install requests",
            1,
        ),
        text.replace(
            f"        run: {PROJECT_INSTALL}",
            "        run: |\n"
            f"          {PROJECT_INSTALL}\n"
            "          pip install requests",
            1,
        ),
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_immutable_install_contract(mutation)


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


def test_readme_documents_pinned_dataset_and_fail_closed_scoring() -> None:
    text = README.read_text(encoding="utf-8")

    assert "c104f840cc67f8b6eec6f759ebc8b2693d585d4a" in text
    assert "f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c" in text
    assert "8 MiB per file" in text
    assert "16 MiB per instance bundle" in text
    assert "80 * official_resolved / requested" in text
    assert "10 * non_infrastructure_instances / requested" in text
    assert "5 * explicit_agreements / official_terminal_instances" in text
    assert "5 * completed_within_budget_instances / requested" in text
    assert "Missing or invalid verdict/telemetry receives no credit" in text
