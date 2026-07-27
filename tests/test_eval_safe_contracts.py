from copy import deepcopy

import pytest

from eval import safe_contracts
from eval.safe_contracts import (
    has_verified_coverage_proof,
    normalize_official_resolved,
    sanitize_output_text,
    validate_safe_coverage_proof,
)


def safe_coverage_proof() -> dict:
    return {
        "source": "generated",
        "status": "generated_verified",
        "test_files": ["tests/test_auth.py"],
        "fixed_runs": [
            {
                "outcome": "pass",
                "failing_test_ids": [],
                "assertion_fingerprint": "",
            },
            {
                "outcome": "pass",
                "failing_test_ids": [],
                "assertion_fingerprint": "",
            },
        ],
        "base_runs": [
            {
                "outcome": "assertion_failure",
                "failing_test_ids": ["tests/test_auth.py::test_login"],
                "assertion_fingerprint": "a" * 64,
            },
            {
                "outcome": "assertion_failure",
                "failing_test_ids": ["tests/test_auth.py::test_login"],
                "assertion_fingerprint": "a" * 64,
            },
        ],
    }


def test_safe_coverage_contract_accepts_complete_canonical_proof():
    proof = safe_coverage_proof()

    assert validate_safe_coverage_proof(proof) == proof
    assert has_verified_coverage_proof("generated_verified", proof) is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda proof: proof.update(status="existing_verified"),
        lambda proof: proof.update(extra_payload="raw evaluator output"),
        lambda proof: proof.pop("base_runs"),
        lambda proof: proof["fixed_runs"][0].update(
            failing_test_ids=["tests/test_auth.py::test_login"]
        ),
        lambda proof: proof["fixed_runs"][0].update(
            assertion_fingerprint="b" * 64
        ),
        lambda proof: proof["base_runs"][0].update(failing_test_ids=[]),
        lambda proof: proof["base_runs"][0].update(
            assertion_fingerprint="not-a-sha256"
        ),
        lambda proof: proof["base_runs"][1].update(
            assertion_fingerprint="b" * 64
        ),
        lambda proof: proof["base_runs"][1].update(
            failing_test_ids=["tests/test_auth.py::test_other"]
        ),
        lambda proof: proof["fixed_runs"][0].update(raw_logs="secret"),
    ],
)
def test_safe_coverage_contract_rejects_malformed_structure(mutate):
    proof = deepcopy(safe_coverage_proof())
    mutate(proof)

    assert validate_safe_coverage_proof(proof) is None
    assert has_verified_coverage_proof("generated_verified", proof) is False


@pytest.mark.parametrize(
    "test_files",
    [
        ["../tests/test_auth.py"],
        ["tests//test_auth.py"],
        ["tests\\test_auth.py"],
        ["/tests/test_auth.py"],
        [".github/tests/test_auth.py"],
        ["vendor/tests/test_auth.py"],
        ["tests/test_gold_patch.py"],
        ["src/auth.py"],
        ["tests/test_auth.py", "tests/test_auth.py"],
    ],
)
def test_safe_coverage_contract_rejects_non_test_or_forbidden_paths(test_files):
    proof = safe_coverage_proof()
    proof["test_files"] = test_files
    proof["base_runs"][0]["failing_test_ids"] = [f"{test_files[0]}::test_login"]
    proof["base_runs"][1]["failing_test_ids"] = [f"{test_files[0]}::test_login"]

    assert validate_safe_coverage_proof(proof) is None


def test_safe_coverage_contract_requires_failure_ids_to_bind_test_files():
    proof = safe_coverage_proof()
    for run in proof["base_runs"]:
        run["failing_test_ids"] = ["tests/test_other.py::test_login"]

    assert validate_safe_coverage_proof(proof) is None


def test_safe_coverage_contract_rejects_control_characters_in_test_id():
    proof = safe_coverage_proof()
    for run in proof["base_runs"]:
        run["failing_test_ids"] = ["tests/test_auth.py::test_login\ninjected"]

    assert validate_safe_coverage_proof(proof) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
        ("true", None),
        ("false", None),
        (1, None),
        (0, None),
    ],
)
def test_official_resolved_accepts_only_actual_booleans(value, expected):
    assert normalize_official_resolved(value) is expected


@pytest.mark.parametrize(
    "error_class",
    ["ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"],
)
def test_exception_class_normalizer_preserves_gateway_timeout_classes(
    error_class: str,
) -> None:
    exception = type(error_class, (TimeoutError,), {})()

    assert safe_contracts.normalize_exception_class(exception) == error_class
    assert safe_contracts.normalize_exception_class(error_class) == error_class


@pytest.mark.parametrize(
    "value",
    ["CredentialSentinel", "ReadTimeout secret-token", "api_key=secret-value"],
)
def test_exception_class_normalizer_rejects_arbitrary_or_secret_bearing_values(
    value: str,
) -> None:
    assert safe_contracts.normalize_exception_class(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Bearer plain-secret-987654321", "Bearer [REDACTED]"),
        ("Authorization: plain-secret-987654321", "Authorization: [REDACTED]"),
        ("x-api-key: plain-secret-987654321", "x-api-key: [REDACTED]"),
        ("api_key=plain-secret-987654321", "api_key=[REDACTED]"),
        ("api-key: plain-secret-987654321", "api-key: [REDACTED]"),
        ("token=plain-secret-987654321", "token=[REDACTED]"),
        ("access_token=plain-secret-987654321", "access_token=[REDACTED]"),
        (
            "https://example.test/?access_token=plain-secret-987654321&ok=1",
            "https://example.test/?access_token=[REDACTED]&ok=1",
        ),
        (
            '{"api_key": "plain-secret-987654321", "ok": true}',
            '{"api_key": "[REDACTED]", "ok": true}',
        ),
        (
            "access_token: 'plain-secret-987654321' # keep-comment",
            "access_token: '[REDACTED]' # keep-comment",
        ),
        (
            'API_KEY="plain-secret-987654321"',
            'API_KEY="[REDACTED]"',
        ),
    ],
)
def test_output_sanitizer_redacts_plain_secret_forms_without_breaking_structure(
    value, expected
):
    assert sanitize_output_text(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "LLM_API_KEY=plain-secret-987654321",
            "LLM_API_KEY=[REDACTED]",
        ),
        (
            'LLM_ESCALATION_API_KEY = "plain-secret-987654321" # primary',
            'LLM_ESCALATION_API_KEY = "[REDACTED]" # primary',
        ),
        (
            "GITHUB_TOKEN='plain-secret-987654321' # release",
            "GITHUB_TOKEN='[REDACTED]' # release",
        ),
        (
            "x_api_key = plain-secret-987654321 # lower-case",
            "x_api_key = [REDACTED] # lower-case",
        ),
        (
            "repo_AUTHORIZATION = 'plain-secret-987654321'",
            "repo_AUTHORIZATION = '[REDACTED]'",
        ),
        (
            'SERVICE_ACCESS_TOKEN="plain-secret-987654321"',
            'SERVICE_ACCESS_TOKEN="[REDACTED]"',
        ),
        (
            "export Mixed_Case_Api_Key = 'plain-secret-987654321' # keep",
            "export Mixed_Case_Api_Key = '[REDACTED]' # keep",
        ),
    ],
)
def test_output_sanitizer_redacts_prefixed_secret_env_assignments(
    value, expected
):
    assert sanitize_output_text(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "LLM_MODEL=plain-secret-987654321",
        "TOKENIZER_MODEL=plain-secret-987654321",
        "ACCESS_TOKEN_TTL=plain-secret-987654321",
        "AUTHORIZATION_MODE=plain-secret-987654321",
        "API_KEY_HINT=plain-secret-987654321",
    ],
)
def test_output_sanitizer_does_not_redact_non_sensitive_env_names(value):
    assert sanitize_output_text(value) == value
