from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval import oci_contract, safe_contracts
from eval.oci_contract import (
    REPO_ROOT,
    InstanceManifest,
    OfficialResult,
    RuntimeRecord,
    load_mode_instance_ids,
    require_mode_instance,
    sha256_file,
    write_model,
)

COMMIT_SHA = "a" * 40
IMAGE_SHA = "sha256:" + "b" * 64
ROW_SHA = "c" * 64
INSTANCE_ID = "pytest-dev__pytest-10081"
APPROVED_MODEL_INVOCATION_NODES = (
    "plan",
    "plan_fix",
    "reflect",
    "reflect_on_failure",
    "test_generation",
    "test_generator",
    "outcome_summary",
)


def _invocation_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "gemini-3.5-flash:stable",
        "provider": "primary",
        "node": "plan",
        "elapsed_seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "status": "ok",
        "error_class": "",
    }
    payload.update(overrides)
    return payload


def _coverage_proof() -> dict[str, object]:
    passing_run = {
        "outcome": "pass",
        "failing_test_ids": [],
        "assertion_fingerprint": "",
    }
    failing_run = {
        "outcome": "assertion_failure",
        "failing_test_ids": ["tests/test_auth.py::test_login"],
        "assertion_fingerprint": "f" * 64,
    }
    return {
        "source": "generated",
        "status": "generated_verified",
        "test_files": ["tests/test_auth.py"],
        "fixed_runs": [dict(passing_run), dict(passing_run)],
        "base_runs": [dict(failing_run), dict(failing_run)],
    }


def _result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": INSTANCE_ID,
        "mode": "agent_v2",
        "evaluation_mode": "end_to_end",
        "model": "gemini-3.5-flash:stable",
        "commit_sha": COMMIT_SHA,
        "repo": "pytest-dev/pytest",
        "issue_url": "https://github.com/pytest-dev/pytest/issues/10081",
        "issue_title": "A regression",
        "success": False,
        "agent_success": False,
        "official_resolved": None,
        "waiting_for_user": False,
        "final_phase": "FAILED",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "turns_taken": 2,
        "token_used": 15,
        "error": None,
        "replay": None,
        "replay_error": None,
        "models_used": ["gemini-3.5-flash:stable"],
        "escalated": False,
        "escalation_reason": "",
        "model_invocations": [_invocation_payload()],
        "tool_invocations": [],
        "unique_evidence_count": 0,
        "max_consecutive_no_progress": 0,
        "attempt_outcome_summary": "tests failed",
        "coverage_status": "failed",
        "coverage_test_files": [],
        "coverage_test_command": "",
        "coverage_proof": None,
        "coverage_failure_reason": "test_failed",
        "test_generation_attempts": 0,
        "failure_class": "test_failed",
        "instance_id": INSTANCE_ID,
        "base_commit": "d" * 40,
        "model_patch": "",
    }
    payload.update(overrides)
    return payload


def _result_record_model():
    model = getattr(oci_contract, "ResultRecord", None)
    assert model is not None, "ResultRecord contract is missing"
    return model


def _manifest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": "checkpoint_5",
        "instance_id": INSTANCE_ID,
        "commit_sha": COMMIT_SHA,
        "row_sha256": ROW_SHA,
        "runtime_status": "ready",
        "image_sha": IMAGE_SHA,
        "primary_model": "gemini-3.5-flash:stable",
        "escalation_model": "claude-opus-4-8:stable",
        "files": {
            "result.json": "1" * 64,
            "prediction.jsonl": "2" * 64,
            "official_result.json": "3" * 64,
        },
    }
    payload.update(overrides)
    return payload


def test_mode_ids_preserve_tracked_order(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "checkpoint_5_ids.txt").write_text("b\na\n", encoding="utf-8")

    assert load_mode_instance_ids("checkpoint_5", tmp_path) == ("b", "a")


def test_baseline_50_is_fixed_unique_and_preserves_historical_sets() -> None:
    baseline_50 = load_mode_instance_ids("baseline_50")
    baseline_10 = tuple(
        line.strip()
        for line in (REPO_ROOT / "eval" / "baseline_10_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    checkpoint_5 = load_mode_instance_ids("checkpoint_5")

    assert len(baseline_50) == 50
    assert len(set(baseline_50)) == 50
    assert baseline_50[:10] == baseline_10
    assert set(checkpoint_5) <= set(baseline_50)


def test_retired_baseline_10_is_not_a_public_mode() -> None:
    with pytest.raises(ValueError, match="unsupported evaluation mode"):
        load_mode_instance_ids("baseline_10")


def test_mode_ids_reject_duplicates(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "baseline_50_ids.txt").write_text("a\na\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_mode_instance_ids("baseline_50", tmp_path)


def test_require_mode_instance_rejects_untracked_id(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "checkpoint_5_ids.txt").write_text("allowed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not tracked"):
        require_mode_instance("checkpoint_5", "other", tmp_path)


def test_runtime_record_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="gold_patch"):
        RuntimeRecord(
            mode="checkpoint_5",
            instance_id=INSTANCE_ID,
            commit_sha=COMMIT_SHA,
            status="dataset_infra",
            error_class="DatasetUnavailable",
            gold_patch="must not persist",
        )


def test_ready_runtime_and_manifest_bind_valid_dataset_row_digest() -> None:
    runtime = RuntimeRecord(
        mode="checkpoint_5",
        instance_id=INSTANCE_ID,
        commit_sha=COMMIT_SHA,
        status="ready",
        remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
        image_sha=IMAGE_SHA,
        row_sha256=ROW_SHA,
    )
    manifest = InstanceManifest.model_validate(
        _manifest_payload(row_sha256=ROW_SHA)
    )

    assert runtime.row_sha256 == ROW_SHA
    assert manifest.row_sha256 == ROW_SHA

    with pytest.raises(ValidationError, match="row_sha256"):
        RuntimeRecord(
            mode="checkpoint_5",
            instance_id=INSTANCE_ID,
            commit_sha=COMMIT_SHA,
            status="ready",
            remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
            image_sha=IMAGE_SHA,
        )
    with pytest.raises(ValidationError, match="row_sha256"):
        InstanceManifest.model_validate(_manifest_payload(row_sha256="d" * 63))


def test_manifest_requires_image_sha_only_for_ready_runtime() -> None:
    with pytest.raises(ValidationError, match="image_sha"):
        InstanceManifest.model_validate(_manifest_payload(image_sha=""))

    manifest = InstanceManifest.model_validate(
        _manifest_payload(runtime_status="oci_image_infra", image_sha="")
    )

    assert manifest.image_sha == ""


def test_manifest_rejects_image_claim_for_failed_runtime() -> None:
    with pytest.raises(ValidationError, match="image_sha"):
        InstanceManifest.model_validate(
            _manifest_payload(runtime_status="oci_boundary_infra")
        )


def test_manifest_requires_exact_safe_file_set() -> None:
    files = dict(_manifest_payload()["files"])
    files["raw.log"] = "4" * 64

    with pytest.raises(ValidationError, match="files"):
        InstanceManifest.model_validate(_manifest_payload(files=files))


def test_official_result_distinguishes_unresolved_from_infrastructure() -> None:
    unresolved = OfficialResult(
        instance_id=INSTANCE_ID,
        status="unresolved",
        submitted=True,
        completed=True,
        resolved=False,
    )
    infrastructure = OfficialResult(
        instance_id=INSTANCE_ID,
        status="scorer_infra",
        submitted=False,
        completed=False,
        resolved=False,
        error_class="RuntimeError",
    )

    assert unresolved.status == "unresolved"
    assert infrastructure.status == "scorer_infra"


def test_official_empty_patch_is_submitted_but_not_completed() -> None:
    result = OfficialResult(
        instance_id=INSTANCE_ID,
        status="empty_patch",
        submitted=True,
        completed=False,
        resolved=False,
    )

    assert result.status == "empty_patch"


def test_official_scorer_infrastructure_requires_error_class() -> None:
    with pytest.raises(ValidationError, match="error_class"):
        OfficialResult(
            instance_id=INSTANCE_ID,
            status="scorer_infra",
            submitted=False,
            completed=False,
            resolved=False,
        )


@pytest.mark.parametrize(
    "error_class",
    ["CredentialSentinel", "ReadTimeout secret-token", "api_key=secret-value"],
)
def test_official_scorer_infrastructure_rejects_unsafe_error_class(
    error_class: str,
) -> None:
    with pytest.raises(ValidationError, match="error_class"):
        OfficialResult(
            instance_id=INSTANCE_ID,
            status="scorer_infra",
            submitted=False,
            completed=False,
            resolved=False,
            error_class=error_class,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submitted", 1),
        ("submitted", "true"),
        ("completed", 0),
        ("completed", "false"),
        ("resolved", 1),
        ("resolved", "true"),
    ],
)
def test_official_result_rejects_coerced_boolean_flags(
    field: str,
    value: object,
) -> None:
    payload: dict[str, object] = {
        "instance_id": INSTANCE_ID,
        "status": "resolved",
        "submitted": True,
        "completed": True,
        "resolved": True,
    }
    payload[field] = value

    with pytest.raises(ValidationError, match=field):
        OfficialResult.model_validate(payload)


def test_result_record_accepts_complete_producer_schema() -> None:
    record = _result_record_model().model_validate(_result_payload())

    assert record.instance_id == INSTANCE_ID
    assert record.model_invocations[0].provider == "primary"


@pytest.mark.parametrize("scope", ["result", "invocation"])
def test_result_record_forbids_unknown_fields(scope: str) -> None:
    payload = _result_payload()
    if scope == "result":
        payload["unexpected"] = "unsafe"
    else:
        invocation = dict(payload["model_invocations"][0])
        invocation["unexpected"] = "unsafe"
        payload["model_invocations"] = [invocation]

    with pytest.raises(ValidationError, match="unexpected"):
        _result_record_model().model_validate(payload)


@pytest.mark.parametrize("failure_class", [None, "unknown_failure"])
def test_result_record_requires_known_failure_class(
    failure_class: str | None,
) -> None:
    payload = _result_payload()
    if failure_class is None:
        del payload["failure_class"]
    else:
        payload["failure_class"] = failure_class

    with pytest.raises(ValidationError, match="failure_class"):
        _result_record_model().model_validate(payload)


def test_invocation_status_uses_closed_taxonomy() -> None:
    payload = _result_payload(
        model_invocations=[_invocation_payload(status="garbage")]
    )

    with pytest.raises(ValidationError, match="status"):
        _result_record_model().model_validate(payload)


def test_safe_contract_exposes_exact_model_invocation_node_taxonomy() -> None:
    assert getattr(safe_contracts, "MODEL_INVOCATION_NODES", None) == frozenset(
        APPROVED_MODEL_INVOCATION_NODES
    )


@pytest.mark.parametrize("node", APPROVED_MODEL_INVOCATION_NODES)
def test_invocation_accepts_approved_node(node: str) -> None:
    record = _result_record_model().model_validate(
        _result_payload(model_invocations=[_invocation_payload(node=node)])
    )

    assert record.model_invocations[0].node == node


@pytest.mark.parametrize("node", ["forged-node", "", b"plan", 1, True, None])
def test_invocation_rejects_unknown_empty_or_coerced_node(node: object) -> None:
    with pytest.raises(ValidationError, match="node"):
        _result_record_model().model_validate(
            _result_payload(model_invocations=[_invocation_payload(node=node)])
        )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("primary", "claude-opus-4-8:stable"),
        ("escalation", "gemini-3.5-flash:stable"),
    ],
)
def test_invocation_provider_must_match_configured_model(
    provider: str,
    model: str,
) -> None:
    payload = _result_payload(
        model_invocations=[
            _invocation_payload(provider=provider, model=model)
        ]
    )

    with pytest.raises(ValidationError, match="provider"):
        _result_record_model().model_validate(payload)


def test_invocation_accepts_both_configured_provider_model_pairs() -> None:
    payload = _result_payload(
        models_used=[
            "gemini-3.5-flash:stable",
            "claude-opus-4-8:stable",
        ],
        escalated=True,
        escalation_reason="repeated_no_progress",
        model_invocations=[
            _invocation_payload(),
            _invocation_payload(
                provider="escalation",
                model="claude-opus-4-8:stable",
            ),
        ],
    )

    record = _result_record_model().model_validate(payload)

    assert [item.provider for item in record.model_invocations] == [
        "primary",
        "escalation",
    ]


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        ("ok", "RuntimeError"),
        ("error", ""),
    ],
)
def test_invocation_status_requires_consistent_error_class(
    status: str,
    error_class: str,
) -> None:
    payload = _result_payload(
        model_invocations=[
            _invocation_payload(status=status, error_class=error_class)
        ]
    )

    with pytest.raises(ValidationError, match="error_class"):
        _result_record_model().model_validate(payload)


@pytest.mark.parametrize("error_class", ["", "ValueError"])
def test_invalid_response_allows_sanitized_or_empty_error_class(
    error_class: str,
) -> None:
    record = _result_record_model().model_validate(
        _result_payload(
            model_invocations=[
                _invocation_payload(
                    status="invalid_response",
                    error_class=error_class,
                )
            ]
        )
    )

    assert record.model_invocations[0].error_class == error_class


@pytest.mark.parametrize(
    "error_class",
    ["gateway exploded", "ReadTimeout: secret-token", "api_key=secret-value"],
)
def test_invocation_rejects_unsanitized_error_class(error_class: str) -> None:
    with pytest.raises(ValidationError, match="error_class"):
        _result_record_model().model_validate(
            _result_payload(
                model_invocations=[
                    _invocation_payload(
                        status="error",
                        error_class=error_class,
                    )
                ]
            )
        )


@pytest.mark.parametrize(
    "error_class",
    ["ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"],
)
def test_invocation_accepts_shared_gateway_timeout_class(
    error_class: str,
) -> None:
    record = _result_record_model().model_validate(
        _result_payload(
            model_invocations=[
                _invocation_payload(
                    status="error",
                    error_class=error_class,
                )
            ]
        )
    )

    assert record.model_invocations[0].error_class == error_class


def test_result_models_used_match_ordered_unique_invocation_history() -> None:
    escalation = _invocation_payload(
        provider="escalation",
        model="claude-opus-4-8:stable",
    )
    payload = _result_payload(
        models_used=[
            "claude-opus-4-8:stable",
            "gemini-3.5-flash:stable",
        ],
        escalated=True,
        escalation_reason="repeated_no_progress",
        model_invocations=[
            _invocation_payload(),
            escalation,
            dict(escalation),
        ],
    )

    with pytest.raises(ValidationError, match="models_used"):
        _result_record_model().model_validate(payload)


def test_escalation_invocation_requires_escalated_flag() -> None:
    payload = _result_payload(
        models_used=[
            "gemini-3.5-flash:stable",
            "claude-opus-4-8:stable",
        ],
        model_invocations=[
            _invocation_payload(),
            _invocation_payload(
                provider="escalation",
                model="claude-opus-4-8:stable",
            ),
        ],
    )

    with pytest.raises(ValidationError, match="escalated"):
        _result_record_model().model_validate(payload)


def test_escalated_result_may_precede_first_escalation_invocation() -> None:
    record = _result_record_model().model_validate(
        _result_payload(
            escalated=True,
            escalation_reason="invalid_structured_response_after_retries",
        )
    )

    assert record.escalated is True
    assert [item.provider for item in record.model_invocations] == ["primary"]


def test_empty_invocation_history_uses_primary_model_fallback() -> None:
    record = _result_record_model().model_validate(
        _result_payload(model_invocations=[])
    )

    assert record.models_used == ["gemini-3.5-flash:stable"]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "model_patch": "diff --git a/a.py b/a.py\n",
            "success": False,
            "agent_success": False,
        },
        {
            "model_patch": "diff --git a/a.py b/a.py\n",
            "success": True,
            "agent_success": True,
            "failure_class": "agent_success",
            "coverage_status": "generated_verified",
            "coverage_proof": _coverage_proof(),
            "model_invocations": [],
        },
        {
            "model_patch": "diff --git a/a.py b/a.py\n",
            "success": True,
            "agent_success": True,
            "failure_class": "agent_success",
            "coverage_status": "generated_verified",
            "coverage_proof": _coverage_proof(),
            "model_invocations": [
                _invocation_payload(
                    status="error",
                    error_class="RuntimeError",
                )
            ],
        },
    ],
)
def test_nonempty_patch_requires_successful_model_history(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="model_patch|invocation"):
        _result_record_model().model_validate(_result_payload(**overrides))


@pytest.mark.parametrize(
    "base_commit",
    ["D" * 40, "d" * 39, "not-a-commit"],
)
def test_result_record_rejects_invalid_nonempty_base_commit(
    base_commit: str,
) -> None:
    with pytest.raises(ValidationError, match="base_commit"):
        _result_record_model().model_validate(
            _result_payload(base_commit=base_commit)
        )


def test_result_record_allows_empty_base_commit_for_synthetic_failure() -> None:
    record = _result_record_model().model_validate(
        _result_payload(base_commit="", model_invocations=[])
    )

    assert record.base_commit == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
        ("elapsed_seconds", -1.0),
        ("input_tokens", float("nan")),
        ("input_tokens", float("inf")),
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("input_tokens", True),
    ],
)
def test_invocation_rejects_non_finite_negative_or_coerced_numbers(
    field: str,
    value: object,
) -> None:
    payload = _result_payload(
        model_invocations=[_invocation_payload(**{field: value})]
    )

    with pytest.raises(ValidationError, match=field):
        _result_record_model().model_validate(payload)


def test_result_record_rejects_bool_as_nonnegative_integer() -> None:
    with pytest.raises(ValidationError, match="turns_taken"):
        _result_record_model().model_validate(_result_payload(turns_taken=True))


def test_result_record_requires_matching_result_and_instance_ids() -> None:
    with pytest.raises(ValidationError, match="instance_id"):
        _result_record_model().model_validate(_result_payload(id="other-id"))


def test_agent_success_requires_verified_coverage_proof() -> None:
    successful = {
        "success": True,
        "agent_success": True,
        "failure_class": "agent_success",
        "coverage_status": "generated_verified",
    }
    with pytest.raises(ValidationError, match="coverage"):
        _result_record_model().model_validate(
            _result_payload(**successful, coverage_proof=None)
        )

    record = _result_record_model().model_validate(
        _result_payload(**successful, coverage_proof=_coverage_proof())
    )

    assert record.agent_success is True


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"exact\x00bytes")

    assert sha256_file(path) == hashlib.sha256(b"exact\x00bytes").hexdigest()


def test_write_model_uses_strict_json_schema(tmp_path: Path) -> None:
    path = tmp_path / "official_result.json"
    model = OfficialResult(
        instance_id=INSTANCE_ID,
        status="resolved",
        submitted=True,
        completed=True,
        resolved=True,
    )

    written = write_model(path, model)

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == model.model_dump(
        mode="json"
    )
