"""SWE-bench Verified dataset adapter and prediction serializer."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "main"


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


def load_verified_samples(
    count: int,
    seed: int,
    cache_path: Path | None = None,
    dataset_loader: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Load official rows once, cache them locally, and normalize a selection."""
    path = cache_path or _default_cache_path()
    if path.exists():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    else:
        if dataset_loader is None:
            from datasets import load_dataset

            dataset_loader = load_dataset
        rows = list(
            dataset_loader(
                DATASET_NAME,
                split="test",
                revision=DATASET_REVISION,
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(dict(item), ensure_ascii=False) + "\n" for item in rows
            ),
            encoding="utf-8",
        )
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {"dataset": DATASET_NAME, "revision": DATASET_REVISION},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
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
