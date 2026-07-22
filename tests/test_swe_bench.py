from __future__ import annotations

import json

import pytest

from eval import swe_bench


def _row(
    instance_id: str = "acme__widget-8",
    repo: str = "acme/widget",
) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "issue_id": "7",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "problem_statement": "Login crash\n\nThe endpoint raises ValueError.",
        "base_commit": "a" * 40,
        "patch": "diff --git a/src/auth.py b/src/auth.py\n",
        "test_patch": "diff --git a/tests/test_auth.py b/tests/test_auth.py\n",
        "FAIL_TO_PASS": '["tests/test_auth.py::test_login"]',
        "PASS_TO_PASS": '["tests/test_auth.py::test_other"]',
        "version": "1.0",
        "created_at": "2026-01-01",
        "difficulty": "medium",
    }


def test_normalize_verified_row_preserves_evaluator_fields():
    sample = swe_bench.normalize_verified_row(_row())

    assert sample["id"] == "acme__widget-8"
    assert sample["repo"] == {"owner": "acme", "name": "widget"}
    assert sample["issue"] == {
        "number": 7,
        "url": "https://github.com/acme/widget/issues/7",
        "title": "Login crash",
        "body": "The endpoint raises ValueError.",
    }
    assert sample["base_commit"] == "a" * 40
    assert sample["evaluation"]["fail_to_pass"] == [
        "tests/test_auth.py::test_login"
    ]
    assert sample["evaluation"]["gold_patch"].startswith("diff --git")


def test_agent_seed_excludes_gold_and_test_material():
    sample = swe_bench.normalize_verified_row(_row())

    seed = swe_bench.build_agent_seed(sample, "/tmp/acme-widget")

    encoded = json.dumps(seed)
    assert seed["repo_ref"] == "a" * 40
    assert seed["repo_path"] == "/tmp/acme-widget"
    assert "gold_patch" not in encoded
    assert "test_patch" not in encoded
    assert "fail_to_pass" not in encoded
    assert "pass_to_pass" not in encoded


def test_diverse_selection_is_deterministic_and_spreads_repositories():
    rows = [
        _row("acme__widget-1", "acme/widget"),
        _row("acme__widget-2", "acme/widget"),
        _row("other__tool-3", "other/tool"),
    ]

    first = swe_bench.select_diverse(rows, count=2, seed=17)
    second = swe_bench.select_diverse(rows, count=2, seed=17)

    assert [item["instance_id"] for item in first] == [
        item["instance_id"] for item in second
    ]
    assert {item["repo"] for item in first} == {"acme/widget", "other/tool"}


def test_load_verified_samples_persists_dataset_and_revision(tmp_path):
    calls = []

    def loader(dataset_name, *, split, revision):
        calls.append((dataset_name, split, revision))
        return [_row()]

    cache_path = tmp_path / "verified.jsonl"

    samples = swe_bench.load_verified_samples(
        count=1,
        seed=17,
        cache_path=cache_path,
        dataset_loader=loader,
    )

    assert samples[0]["instance_id"] == "acme__widget-8"
    assert calls == [
        (swe_bench.DATASET_NAME, "test", swe_bench.DATASET_REVISION)
    ]
    assert json.loads(cache_path.with_suffix(".metadata.json").read_text()) == {
        "dataset": swe_bench.DATASET_NAME,
        "revision": swe_bench.DATASET_REVISION,
    }


def test_write_predictions_uses_official_schema_without_evaluator_data(tmp_path):
    results = [
        {
            "instance_id": "acme__widget-8",
            "model": "gemini-3.5-flash:stable",
            "model_patch": "diff --git a/src/auth.py b/src/auth.py\n",
            "evaluation": {"gold_patch": "secret gold patch"},
        }
    ]
    output_path = tmp_path / "predictions.jsonl"

    written = swe_bench.write_predictions(results, output_path)

    payload = json.loads(written.read_text().strip())
    assert payload == {
        "instance_id": "acme__widget-8",
        "model_name_or_path": "gemini-3.5-flash:stable",
        "model_patch": "diff --git a/src/auth.py b/src/auth.py\n",
    }
    assert "secret gold patch" not in written.read_text()


def test_write_predictions_preserves_previous_checkpoint_when_replace_fails(
    monkeypatch, tmp_path
):
    output_path = tmp_path / "predictions.jsonl"
    previous = (
        '{"instance_id":"old","model_name_or_path":"old-model",'
        '"model_patch":"old-patch"}\n'
    ).encode()
    output_path.write_bytes(previous)

    def fail_replace(source, destination):
        raise OSError("prediction replace interrupted")

    monkeypatch.setattr(swe_bench.os, "replace", fail_replace)

    with pytest.raises(OSError, match="prediction replace interrupted"):
        swe_bench.write_predictions(
            [
                {
                    "instance_id": "acme__widget-8",
                    "model": "test-model",
                    "model_patch": "new-patch",
                }
            ],
            output_path,
        )

    assert output_path.read_bytes() == previous
    assert list(tmp_path.glob(".*.tmp")) == []
