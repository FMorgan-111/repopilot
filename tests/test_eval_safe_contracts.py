from copy import deepcopy

import pytest

from eval.safe_contracts import (
    has_verified_coverage_proof,
    normalize_official_resolved,
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
