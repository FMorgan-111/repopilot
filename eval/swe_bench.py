"""SWE-bench Verified dataset adapter and prediction serializer."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import re
import tempfile
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DATASET_ROW_COUNT = 500
DATASET_CONTENT_SHA256 = (
    "f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c"
)
DATASET_ROW_FIELDS = (
    "repo",
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "created_at",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
    "difficulty",
)
_CACHE_SCHEMA_VERSION = 1


def _validated_verified_row(row: Mapping[str, Any]) -> dict[str, str]:
    if set(row) != set(DATASET_ROW_FIELDS):
        raise ValueError("invalid SWE-bench Verified row schema")
    projected = {field: row[field] for field in DATASET_ROW_FIELDS}
    if any(not isinstance(value, str) for value in projected.values()):
        raise ValueError("invalid SWE-bench Verified row schema")
    if (
        not projected["instance_id"]
        or projected["repo"].count("/") != 1
        or not re.fullmatch(r"[0-9a-f]{40}", projected["base_commit"])
        or not re.fullmatch(
            r"[0-9a-f]{40}", projected["environment_setup_commit"]
        )
    ):
        raise ValueError("invalid SWE-bench Verified row schema")
    for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        try:
            tests = json.loads(projected[field])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid SWE-bench Verified row schema") from exc
        if not isinstance(tests, list) or any(
            not isinstance(item, str) for item in tests
        ):
            raise ValueError("invalid SWE-bench Verified row schema")
    return projected


def serialize_verified_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize official rows using the code-pinned upstream column order."""
    return b"".join(
        json.dumps(
            _validated_verified_row(row),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def verified_row_sha256(row: Mapping[str, Any]) -> str:
    """Return a stable digest for one validated official dataset row."""
    serialized = serialize_verified_rows([row])
    return hashlib.sha256(serialized.removesuffix(b"\n")).hexdigest()


def _expected_cache_metadata() -> dict[str, Any]:
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "split": "test",
        "row_count": DATASET_ROW_COUNT,
        "content_sha256": DATASET_CONTENT_SHA256,
    }


def _validate_dataset_contents(contents: bytes) -> list[dict[str, str]]:
    if not hmac.compare_digest(
        hashlib.sha256(contents).hexdigest(), DATASET_CONTENT_SHA256
    ):
        raise ValueError("SWE-bench Verified content SHA-256 mismatch")
    try:
        decoded = contents.decode("utf-8")
        rows = [json.loads(line) for line in decoded.splitlines() if line]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid SWE-bench Verified cache content") from exc
    if len(rows) != DATASET_ROW_COUNT:
        raise ValueError("invalid SWE-bench Verified row count")
    validated = [_validated_verified_row(row) for row in rows]
    if len({row["instance_id"] for row in validated}) != len(validated):
        raise ValueError("duplicate SWE-bench Verified instance ID")
    if serialize_verified_rows(validated) != contents:
        raise ValueError("non-canonical SWE-bench Verified cache content")
    return validated


def atomic_write_text(path: Path, contents: str) -> Path:
    """Durably replace a text file without truncating its prior contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return path


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    decoded = json.loads(value)
    return [str(item) for item in decoded]


def normalize_verified_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map an official SWE-bench row to RepoPilot's evaluation contract."""
    owner, name = str(row["repo"]).split("/", 1)
    statement = str(row.get("problem_statement") or "")
    first_line, _, rest = statement.partition("\n")
    return {
        "id": str(row["instance_id"]),
        "instance_id": str(row["instance_id"]),
        "source": "swe-bench-verified",
        "repo": {"owner": owner, "name": name},
        "issue": {
            "number": int(row.get("issue_id") or 0),
            "url": str(row.get("issue_url") or ""),
            "title": first_line.strip() or str(row["instance_id"]),
            "body": rest.strip() or statement,
        },
        "base_commit": str(row["base_commit"]),
        "evaluation": {
            "gold_patch": str(row.get("patch") or ""),
            "test_patch": str(row.get("test_patch") or ""),
            "fail_to_pass": _json_list(row.get("FAIL_TO_PASS")),
            "pass_to_pass": _json_list(row.get("PASS_TO_PASS")),
        },
        "metadata": {
            "version": row.get("version"),
            "created_at": row.get("created_at"),
            "difficulty": row.get("difficulty"),
        },
    }


def select_diverse(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Mapping[str, Any]]:
    """Select deterministically while round-robining repositories."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    grouped: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    for item in shuffled:
        grouped[str(item["repo"])].append(item)
    repos = list(grouped)
    rng.shuffle(repos)

    selected: list[Mapping[str, Any]] = []
    while repos and len(selected) < count:
        remaining: list[str] = []
        for repo in repos:
            selected.append(grouped[repo].popleft())
            if grouped[repo]:
                remaining.append(repo)
            if len(selected) == count:
                break
        repos = remaining
    return selected


def _default_cache_path() -> Path:
    root = Path(os.getenv("REPOPILOT_HOME", Path.home() / ".repopilot"))
    return root / "eval" / "datasets" / "swe-bench-verified.jsonl"


def load_verified_rows(
    cache_path: Path | None = None,
    dataset_loader: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Load and cache the exact official SWE-bench Verified rows."""
    path = cache_path or _default_cache_path()
    if path.exists():
        metadata_path = path.with_suffix(".metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid SWE-bench Verified cache metadata") from exc
        if metadata != _expected_cache_metadata():
            raise ValueError("invalid SWE-bench Verified cache metadata")
        return _validate_dataset_contents(path.read_bytes())
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    rows = [
        _validated_verified_row(dict(item))
        for item in dataset_loader(
            DATASET_NAME,
            split="test",
            revision=DATASET_REVISION,
        )
    ]
    contents = serialize_verified_rows(rows)
    rows = _validate_dataset_contents(contents)
    atomic_write_text(path, contents.decode("utf-8"))
    atomic_write_text(
        path.with_suffix(".metadata.json"),
        json.dumps(
            _expected_cache_metadata(),
            sort_keys=True,
        ),
    )
    return rows


def load_verified_instance(
    instance_id: str,
    cache_path: Path | None = None,
    dataset_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return exactly one official row and reject ambiguous dataset state."""
    matches = [
        row
        for row in load_verified_rows(cache_path, dataset_loader)
        if str(row.get("instance_id")) == instance_id
    ]
    if not matches:
        raise ValueError(
            f"unknown SWE-bench Verified instance ID: {instance_id}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"duplicate SWE-bench Verified instance ID: {instance_id}"
        )
    return matches[0]


def load_verified_samples(
    count: int,
    seed: int,
    cache_path: Path | None = None,
    dataset_loader: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Load official rows once, cache them locally, and normalize a selection."""
    rows = load_verified_rows(cache_path, dataset_loader)
    return [
        normalize_verified_row(item)
        for item in select_diverse(rows, count=count, seed=seed)
    ]


def build_agent_seed(
    sample: Mapping[str, Any], repo_path: str
) -> dict[str, Any]:
    """Build the allowlisted benchmark data visible to RepoPilot."""
    return {
        "owner": sample["repo"]["owner"],
        "repo": sample["repo"]["name"],
        "issue_number": sample["issue"]["number"],
        "issue_title": sample["issue"]["title"],
        "issue_body": sample["issue"]["body"],
        "repo_ref": sample["base_commit"],
        "repo_path": repo_path,
    }


def write_predictions(
    results: Sequence[Mapping[str, Any]], path: Path
) -> Path:
    """Write only the schema accepted by the official SWE-bench harness."""
    rows = [
        {
            "instance_id": item["instance_id"],
            "model_name_or_path": item["model"],
            "model_patch": item.get("model_patch", ""),
        }
        for item in results
    ]
    atomic_write_text(
        path,
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
    )
    return path
