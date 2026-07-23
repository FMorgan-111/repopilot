"""Agent-v2 eval runner with saved-run replay diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

agent_v2 = importlib.import_module("src.new_agent").agent_v2
replay_run = importlib.import_module("src.run_store").replay_run
load_run = importlib.import_module("src.run_store").load_run
close_llm_client = importlib.import_module("src.http_client").close_llm_client
get_llm_model = importlib.import_module("src.http_client")._get_llm_model
_model_provider = importlib.import_module("src.model_provider")
escalation_is_configured = _model_provider.escalation_is_configured
redact_secrets = _model_provider.redact_secrets
_evaluator_safety = importlib.import_module("src.evaluator_safety")
safe_prediction_patch = _evaluator_safety.safe_prediction_patch
sanitize_evaluator_text = _evaluator_safety.sanitize_evaluator_text
repopilot_home = importlib.import_module("src.home").repopilot_home
close_store = importlib.import_module("src.memory").close_store
AgentState = importlib.import_module("src.state").AgentState
Tracer = importlib.import_module("src.tracer").Tracer
CoverageProof = importlib.import_module("src.state").CoverageProof
DEFAULT_AGENT_V2_TOKEN_BUDGET = importlib.import_module(
    "src.state"
).DEFAULT_AGENT_V2_TOKEN_BUDGET
_execute = importlib.import_module("src.nodes.execute")
git_clone = _execute.git_clone
is_transient_git_transport_error = _execute._is_transient_git_transport_error
classify_sample = importlib.import_module("eval.failure_taxonomy").classify_sample
_swe_bench = importlib.import_module("eval.swe_bench")
build_agent_seed = _swe_bench.build_agent_seed
load_verified_instance = _swe_bench.load_verified_instance
load_verified_samples = _swe_bench.load_verified_samples
normalize_verified_row = _swe_bench.normalize_verified_row
write_predictions = _swe_bench.write_predictions
atomic_write_text = _swe_bench.atomic_write_text
_safe_contracts = importlib.import_module("eval.safe_contracts")
has_verified_coverage_proof = _safe_contracts.has_verified_coverage_proof
safe_int = _safe_contracts.safe_int
safe_float = _safe_contracts.safe_float
sanitize_output_text = _safe_contracts.sanitize_output_text
validate_safe_coverage_proof = _safe_contracts.validate_safe_coverage_proof
minimal_subprocess_env = importlib.import_module(
    "src.safe_subprocess"
).minimal_subprocess_env

SAMPLES_PATH = REPO_ROOT / "data" / "samples" / "issues_fixes.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "eval_results.json"
MAX_SAMPLES = 5
_ARCHIVE_DIFF_RE = re.compile(
    r"(?im)^diff --git a/\S+\.(?:7z|bz2|egg|gz|rar|tar|tgz|whl|xz|zip) "
)


def _configured_model() -> str:
    return get_llm_model()


def _safe_text(value: Any, limit: int = 2_000) -> str:
    sanitized = sanitize_evaluator_text(redact_secrets(str(value or "")))
    return sanitize_output_text(sanitized, limit)


def _safe_int(value: Any, default: int = 0) -> int:
    return safe_int(value, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return safe_float(value, default)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _sequence_from_state_or_payload(
    state: Any, state_name: str, payload: dict[str, Any], payload_name: str
) -> list[Any]:
    state_values = list(getattr(state, state_name, []) or []) if state is not None else []
    return state_values or list(payload.get(payload_name) or [])


def _safe_model_invocations(values: list[Any]) -> list[dict[str, Any]]:
    allowed = (
        "model",
        "provider",
        "node",
        "elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "status",
        "error_class",
    )
    invocations = []
    for value in values:
        item = _as_mapping(value)
        if not item:
            continue
        safe = {key: item.get(key) for key in allowed}
        for key in ("model", "provider", "node", "status", "error_class"):
            safe[key] = _safe_text(safe.get(key), 300)
        safe["elapsed_seconds"] = _safe_float(safe["elapsed_seconds"])
        safe["input_tokens"] = max(0, _safe_int(safe["input_tokens"]))
        safe["output_tokens"] = max(0, _safe_int(safe["output_tokens"]))
        invocations.append(safe)
    return invocations


def _safe_tool_invocations(values: list[Any]) -> list[dict[str, Any]]:
    allowed = (
        "action",
        "args_fingerprint",
        "status",
        "evidence_id",
        "error_class",
    )
    invocations = []
    for value in values:
        item = _as_mapping(value)
        if not item:
            continue
        safe = {key: item.get(key) for key in allowed}
        for key in allowed:
            if safe[key] is not None:
                safe[key] = _safe_text(safe[key], 300)
        invocations.append(safe)
    return invocations


def _safe_coverage_run(value: Any) -> dict[str, Any]:
    run = _as_mapping(value)
    failing_ids = [
        _safe_text(item, 500)
        for item in run.get("failing_test_ids") or []
        if _safe_text(item, 500)
    ][:100]
    return {
        "outcome": _safe_text(run.get("outcome"), 40),
        "failing_test_ids": failing_ids,
        "assertion_fingerprint": _safe_text(
            run.get("assertion_fingerprint"), 128
        ),
    }


def _safe_coverage_proof(value: Any) -> dict[str, Any] | None:
    try:
        validated = CoverageProof.model_validate(_as_mapping(value))
    except (TypeError, ValueError):
        return None
    proof = validated.model_dump(mode="json")
    projected = {
        "source": _safe_text(proof.get("source"), 40),
        "status": _safe_text(proof.get("status"), 40),
        "test_files": [
            _safe_text(item, 500)
            for item in proof.get("test_files") or []
            if _safe_text(item, 500)
        ][:16],
        "fixed_runs": [
            _safe_coverage_run(item) for item in (proof.get("fixed_runs") or [])[:2]
        ],
        "base_runs": [
            _safe_coverage_run(item) for item in (proof.get("base_runs") or [])[:2]
        ],
    }
    return validate_safe_coverage_proof(projected)


def _max_consecutive_no_progress(values: list[Any], current: int = 0) -> int:
    maximum = max(0, _safe_int(current))
    previous: tuple[str, str, str] | None = None
    streak = 0
    for value in values:
        item = _as_mapping(value)
        identity = (
            _safe_text(item.get("kind"), 100),
            _safe_text(item.get("fingerprint"), 128),
            _safe_text(item.get("node"), 100),
        )
        streak = streak + 1 if identity == previous else 1
        maximum = max(maximum, streak)
        previous = identity
    return maximum


def _safe_model_patch(value: Any) -> str:
    patch = str(value or "")
    if safe_prediction_patch(patch) != patch:
        return ""
    if redact_secrets(patch) != patch or _ARCHIVE_DIFF_RE.search(patch):
        return ""
    return patch


def _safe_replay(value: Any) -> dict[str, Any] | None:
    replay = _as_mapping(value)
    if not replay:
        return None
    safe: dict[str, Any] = {
        "run_id": _safe_text(replay.get("run_id"), 300),
        "issue_url": _safe_text(replay.get("issue_url"), 2_000),
        "current_phase": _safe_text(replay.get("current_phase"), 100),
        "attempt_outcome_summary": _safe_text(
            replay.get("attempt_outcome_summary"), 200
        ),
        "timeline": [],
    }
    for raw_item in replay.get("timeline") or []:
        item = _as_mapping(raw_item)
        item_type = item.get("type")
        if item_type == "decision_frame":
            selected = _as_mapping(item.get("selected_hypothesis"))
            route = _as_mapping(item.get("route"))
            safe["timeline"].append(
                {
                    "index": _safe_int(item.get("index")),
                    "type": "decision_frame",
                    "frame_id": _safe_text(item.get("frame_id"), 300),
                    "stage": _safe_text(item.get("stage"), 100),
                    "summary": _safe_text(item.get("summary"), 1_000),
                    "selected_hypothesis_id": _safe_text(
                        item.get("selected_hypothesis_id"), 300
                    ),
                    "selected_hypothesis": {
                        "id": _safe_text(selected.get("id"), 300),
                        "claim": _safe_text(selected.get("claim"), 1_000),
                    },
                    "recommended_action": _safe_text(
                        item.get("recommended_action"), 100
                    ),
                    "risk": _safe_text(item.get("risk"), 50),
                    "confidence": _safe_float(item.get("confidence")),
                    "route": {"route": _safe_text(route.get("route"), 100)},
                    "warnings": [
                        {
                            "expected_phase": _safe_text(
                                _as_mapping(warning).get("expected_phase"), 100
                            ),
                            "actual_phase": _safe_text(
                                _as_mapping(warning).get("actual_phase"), 100
                            ),
                        }
                        for warning in (item.get("warnings") or [])[:20]
                    ],
                    "next_checks": [
                        _safe_text(check, 500)
                        for check in (item.get("next_checks") or [])[:20]
                    ],
                }
            )
        elif item_type == "node_diagnostic":
            diagnostic = _as_mapping(item.get("diagnostic") or item)
            safe["timeline"].append(
                {
                    "index": _safe_int(item.get("index")),
                    "type": "node_diagnostic",
                    "diagnostic": {
                        key: _safe_text(diagnostic.get(key), 500)
                        for key in (
                            "node",
                            "event",
                            "status",
                            "error_type",
                            "error",
                            "phase_timeout_seconds",
                        )
                        if diagnostic.get(key) is not None
                    },
                }
            )
    return safe


def _safe_result_fields(
    payload: dict[str, Any], state: Any, replay: dict[str, Any] | None
) -> dict[str, Any]:
    model_values = _sequence_from_state_or_payload(
        state, "model_history", payload, "model_history"
    )
    tool_values = _sequence_from_state_or_payload(
        state, "tool_history", payload, "tool_history"
    )
    evidence_values = _sequence_from_state_or_payload(
        state, "evidence", payload, "evidence"
    )
    no_progress_values = _sequence_from_state_or_payload(
        state, "no_progress_history", payload, "no_progress_history"
    )
    model_invocations = _safe_model_invocations(model_values)
    models_used = list(
        dict.fromkeys(
            item["model"] for item in model_invocations if item.get("model")
        )
    )
    if not models_used:
        models_used = [_safe_text(_configured_model(), 300)]

    fingerprints = {
        _safe_text(_as_mapping(item).get("fingerprint"), 128)
        for item in evidence_values
    }
    fingerprints.discard("")
    coverage_status = _safe_text(
        payload.get(
            "coverage_status",
            getattr(state, "coverage_status", "pending") if state is not None else "pending",
        ),
        40,
    )
    if coverage_status not in {
        "pending",
        "existing_verified",
        "generated_verified",
        "failed",
    }:
        coverage_status = "pending"
    raw_proof = payload.get(
        "coverage_proof",
        getattr(state, "coverage_proof", None) if state is not None else None,
    )
    coverage_proof = _safe_coverage_proof(raw_proof)
    summary = payload.get("attempt_outcome_summary")
    if summary is None and state is not None:
        summary = getattr(state, "attempt_outcome_summary", "")
    if not summary and replay:
        summary = replay.get("attempt_outcome_summary", "")
    current_no_progress = payload.get(
        "no_progress_rounds",
        getattr(state, "no_progress_rounds", 0) if state is not None else 0,
    )
    return {
        "models_used": models_used,
        "escalated": bool(payload.get("escalated", getattr(state, "escalated", False))),
        "escalation_reason": _safe_text(
            payload.get(
                "escalation_reason",
                getattr(state, "escalation_reason", "") if state is not None else "",
            ),
            100,
        ),
        "model_invocations": model_invocations,
        "tool_invocations": _safe_tool_invocations(tool_values),
        "unique_evidence_count": len(fingerprints),
        "max_consecutive_no_progress": _max_consecutive_no_progress(
            no_progress_values, _safe_int(current_no_progress)
        ),
        "attempt_outcome_summary": _safe_text(summary, 200),
        "coverage_status": coverage_status,
        "coverage_test_files": [
            _safe_text(item, 500)
            for item in payload.get(
                "coverage_test_files",
                getattr(state, "coverage_test_files", []) if state is not None else [],
            )
            or []
        ][:16],
        "coverage_test_command": _safe_text(
            payload.get(
                "coverage_test_command",
                getattr(state, "coverage_test_command", "") if state is not None else "",
            ),
            2_000,
        ),
        "coverage_proof": coverage_proof,
        "coverage_failure_reason": _safe_text(
            payload.get(
                "coverage_failure_reason",
                getattr(state, "coverage_failure_reason", "") if state is not None else "",
            ),
            100,
        ),
        "test_generation_attempts": max(
            0,
            _safe_int(
                payload.get(
                    "test_generation_attempts",
                    getattr(state, "test_generation_attempts", 0)
                    if state is not None
                    else 0,
                )
            ),
        ),
    }


def _current_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            env=minimal_subprocess_env(),
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _fallback_results_path() -> Path:
    return repopilot_home() / "eval" / "eval_results.json"


def _write_results_with_fallback(
    results: list[dict[str, Any]],
    requested_path: Path | str,
) -> Path:
    path = Path(requested_path)
    serialized_results: list[dict[str, Any]] = []
    for value in results:
        result = dict(value)
        if result.get("mode") == "agent_v2":
            result["coverage_proof"] = validate_safe_coverage_proof(
                result.get("coverage_proof")
            )
            verified = has_verified_coverage_proof(
                result.get("coverage_status"), result.get("coverage_proof")
            )
            claimed = result.get("agent_success", result.get("success", False))
            result["agent_success"] = bool(claimed is True and verified)
            result["success"] = result["agent_success"]
        serialized_results.append(result)
    contents = json.dumps(serialized_results, indent=2, ensure_ascii=False)
    try:
        atomic_write_text(path, contents)
        return path
    except OSError as exc:
        fallback_path = _fallback_results_path()
        atomic_write_text(fallback_path, contents)
        print(
            "Warning: failed to write agent v2 eval results to "
            f"{path}: {type(exc).__name__}: {exc}; wrote fallback to {fallback_path}",
            file=sys.stderr,
            flush=True,
        )
        return fallback_path


async def _close_shared_resources() -> None:
    cleanup_steps = [
        ("shared LLM client", close_llm_client),
        ("shared memory store", close_store),
    ]
    for resource_name, close_resource in cleanup_steps:
        try:
            await close_resource()
        except Exception as exc:
            print(
                f"Warning: failed to close {resource_name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def load_samples(
    n: int = MAX_SAMPLES, sample_id: str | None = None,
    samples_path: Path | None = None,
) -> list[dict[str, Any]]:
    # Resolve at call time (not as a default arg) so tests that monkeypatch the
    # module SAMPLES_PATH take effect, while an explicit path still overrides.
    if samples_path is None:
        samples_path = SAMPLES_PATH
    # When a specific sample_id is requested, scan the whole file for it and
    # return just that one (ignores n). Otherwise take the first n lines.
    if sample_id:
        with samples_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("id") == sample_id:
                    return [record]
        raise ValueError(f"sample_id not found in dataset: {sample_id}")

    samples: list[dict[str, Any]] = []
    with samples_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _is_doc_path(path: str) -> bool:
    lower = path.lower()
    return (
        lower.endswith((".rst", ".md", ".txt"))
        or "/docs/" in lower
        or lower.startswith("docs/")
        or "changelog" in lower
        or "/news/" in lower
    )


def _build_gold_seed(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Seed relevant_files from the dataset's known changed files, fetching each
    file's content via the GitHub Contents API (single-file, NOT the flaky code
    search). Removes locate from the critical path so eval measures the patch
    stage, not GitHub search rate-limiting. Returns None if no code file could
    be seeded (fall back to the normal locate path)."""
    from eval.harness import _validate_base_commit, fetch_file_content

    repo = sample["repo"]
    issue = sample["issue"]
    owner, name = repo["owner"], repo["name"]
    base_commit = _validate_base_commit(sample.get("base_commit"))

    files: list[dict[str, Any]] = []
    for entry in sample.get("patch", {}).get("files", []):
        path = entry.get("path", "")
        if not path or _is_doc_path(path):
            continue
        content = fetch_file_content(owner, name, path, base_commit)
        if not content:  # added file with no pre-image, or fetch failed → skip
            continue
        files.append(
            {
                "path": path,
                "content": content,
                "relevance_score": 1.0,
                "reason": "seeded from gold changed files (offline eval)",
            }
        )
    if not files:
        return None
    return {
        "owner": owner,
        "repo": name,
        "issue_number": issue.get("number", 0),
        "issue_title": issue["title"],
        "issue_body": issue["body"],
        "relevant_files": files,
    }


async def _build_eval_seed(
    sample: dict[str, Any], seed_gold_files: bool
) -> dict[str, Any] | None:
    """Build an oracle seed or prepare an exact SWE-bench issue seed."""
    if sample.get("source") != "swe-bench-verified":
        return _build_gold_seed(sample) if seed_gold_files else None
    issue = sample["issue"]
    repo = sample["repo"]
    trace_id = Tracer().trace_id
    prep = AgentState(
        issue_url=issue["url"],
        owner=repo["owner"],
        repo=repo["name"],
        issue_number=issue["number"],
        repo_ref=sample["base_commit"],
        trace_id=trace_id,
        skip_commit=True,
    )
    repo_path = await git_clone(prep)
    seed = build_agent_seed(sample, repo_path)
    seed["trace_id"] = trace_id
    return seed


async def evaluate_agent_v2_sample(
    sample: dict[str, Any],
    idx: int,
    max_retries: int = 3,
    token_budget: int = DEFAULT_AGENT_V2_TOKEN_BUDGET,
    seed_gold_files: bool = False,
) -> dict[str, Any]:
    issue = sample["issue"]
    repo = sample["repo"]
    patch = sample.get("patch", {})
    signals = sample.get("signals", {})
    issue_url = issue["url"]
    is_swe_bench = sample.get("source") == "swe-bench-verified"

    seed = await _build_eval_seed(sample, seed_gold_files)

    payload = await agent_v2(
        issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        save_final_run=True,
        skip_commit=True,
        seed=seed,
    )
    run_id = payload.get("run_id") or payload.get("trace_id") or ""

    replay: dict[str, Any] | None = None
    replay_error: str | None = None
    saved_state: Any = None
    if run_id:
        try:
            replay = replay_run(run_id)
        except Exception as exc:  # replay should not hide the eval result
            replay_error = _safe_text(f"{type(exc).__name__}: {exc}")
        try:
            saved_state = load_run(run_id)
        except Exception:
            # A crashed or mocked run may have no persisted state. The public
            # payload still supplies safe defaults and terminal coverage data.
            saved_state = None

    safe_fields = _safe_result_fields(payload, saved_state, replay)
    terminal_success = bool(
        payload.get("success") is True
        and has_verified_coverage_proof(
            safe_fields["coverage_status"], safe_fields["coverage_proof"]
        )
    )

    result = {
        "id": _safe_text(sample["id"], 500),
        "mode": "agent_v2",
        "evaluation_mode": (
            "oracle_files" if seed_gold_files and not is_swe_bench else "end_to_end"
        ),
        "model": _safe_text(_configured_model(), 300),
        "commit_sha": _current_commit_sha(),
        "repo": _safe_text(f"{repo['owner']}/{repo['name']}", 500),
        "issue_url": _safe_text(issue_url),
        "issue_title": _safe_text(issue["title"]),
        "success": terminal_success,
        "agent_success": terminal_success,
        # Filled only by an official scorer import; never inferred internally.
        "official_resolved": None,
        "waiting_for_user": payload.get("waiting_for_user") is True,
        "final_phase": _safe_text(payload.get("final_phase", ""), 100),
        "run_id": _safe_text(run_id, 300),
        "trace_id": _safe_text(payload.get("trace_id", ""), 300),
        "turns_taken": max(0, _safe_int(payload.get("turns_taken"))),
        "token_used": max(0, _safe_int(payload.get("token_used"))),
        "error": _safe_text(payload.get("error")) or None,
        "replay": _safe_replay(replay),
        "replay_error": replay_error,
        **safe_fields,
    }
    if not is_swe_bench:
        result.update(
            {
                "actual_files": [
                    _safe_text(file.get("path"), 500)
                    for file in patch.get("files", [])
                    if isinstance(file, dict) and file.get("path")
                ],
                "has_tests_changed": bool(
                    signals.get("has_tests_changed", False)
                ),
            }
        )

    taxonomy_input = dict(result)
    taxonomy_input["agent_payload"] = payload
    result["failure_class"] = classify_sample(taxonomy_input)["decisive"]
    if is_swe_bench:
        result.update(
            {
                "instance_id": sample["instance_id"],
                "base_commit": sample["base_commit"],
                "model_patch": _safe_model_patch(payload.get("model_patch", "")),
            }
        )
    return result


def safe_failed_sample_result(
    sample: dict[str, Any],
    error: Exception,
    *,
    seed_gold_files: bool = False,
    failure_class: str | None = None,
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Create a safe terminal result without serializing exception text."""
    issue = _as_mapping(sample.get("issue"))
    repo = _as_mapping(sample.get("repo"))
    is_swe_bench = sample.get("source") == "swe-bench-verified"
    if failure_class not in {"infra", "other"}:
        failure_class = (
            "infra" if is_transient_git_transport_error(error) else "other"
        )
    safe_fields = _safe_result_fields({}, None, None)
    safe_fields.update(
        {
            "coverage_status": "failed",
            "coverage_failure_reason": failure_class,
        }
    )
    result = {
        "id": _safe_text(sample.get("id"), 500),
        "mode": "agent_v2",
        "evaluation_mode": (
            "oracle_files" if seed_gold_files and not is_swe_bench else "end_to_end"
        ),
        "model": _safe_text(_configured_model(), 300),
        "commit_sha": _safe_text(commit_sha or _current_commit_sha(), 40),
        "repo": _safe_text(f"{repo.get('owner', '')}/{repo.get('name', '')}", 500),
        "issue_url": _safe_text(issue.get("url")),
        "issue_title": _safe_text(issue.get("title")),
        "success": False,
        "agent_success": False,
        "official_resolved": None,
        "waiting_for_user": False,
        "final_phase": "FAILED",
        "run_id": "",
        "trace_id": "",
        "turns_taken": 0,
        "token_used": 0,
        "error": "Sample evaluation failed.",
        "replay": None,
        "replay_error": None,
        **safe_fields,
        "failure_class": failure_class,
    }
    if is_swe_bench:
        result.update(
            {
                "instance_id": _safe_text(sample.get("instance_id"), 500),
                "base_commit": _safe_text(sample.get("base_commit"), 100),
                "model_patch": "",
            }
        )
    else:
        patch = _as_mapping(sample.get("patch"))
        signals = _as_mapping(sample.get("signals"))
        result.update(
            {
                "actual_files": [
                    _safe_text(_as_mapping(file).get("path"), 500)
                    for file in patch.get("files", [])
                    if _as_mapping(file).get("path")
                ],
                "has_tests_changed": bool(signals.get("has_tests_changed", False)),
            }
        )
    return result


async def run_exact_verified_instance(
    instance_id: str,
    *,
    output_dir: Path,
    max_retries: int = 3,
    token_budget: int = 100_000,
) -> dict[str, Any]:
    """Run exactly one official Verified row and persist one safe checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        row = load_verified_instance(instance_id)
        sample = normalize_verified_row(row)
        try:
            result = await evaluate_agent_v2_sample(
                sample,
                0,
                max_retries=max_retries,
                token_budget=token_budget,
                seed_gold_files=False,
            )
        except Exception as exc:
            result = safe_failed_sample_result(sample, exc)
        _write_results_with_fallback([result], output_dir / "result.json")
        write_predictions([result], output_dir / "prediction.jsonl")
        return result
    finally:
        await _close_shared_resources()


async def run_agent_v2_eval(
    n_samples: int = MAX_SAMPLES,
    max_retries: int = 3,
    token_budget: int = DEFAULT_AGENT_V2_TOKEN_BUDGET,
    results_path: Path | str = RESULTS_PATH,
    sample_id: str | None = None,
    seed_gold_files: bool = False,
    samples_path: Path | None = None,
    dataset: str = "custom",
    dataset_seed: int = 17,
    predictions_path: Path | str | None = None,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        if predictions_path is not None and dataset != "swe-bench-verified":
            raise ValueError(
                "predictions_path requires dataset='swe-bench-verified'"
            )
        if dataset == "swe-bench-verified":
            if sample_ids is None:
                samples = load_verified_samples(n_samples, dataset_seed)
            else:
                available = load_verified_samples(sys.maxsize, dataset_seed)
                duplicate = next(
                    (
                        instance_id
                        for index, instance_id in enumerate(sample_ids)
                        if instance_id in sample_ids[:index]
                    ),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(
                        f"duplicate SWE-bench instance ID: {duplicate}"
                    )
                samples_by_id = {
                    sample["instance_id"]: sample for sample in available
                }
                unknown = next(
                    (
                        instance_id
                        for instance_id in sample_ids
                        if instance_id not in samples_by_id
                    ),
                    None,
                )
                if unknown is not None:
                    raise ValueError(f"unknown SWE-bench instance ID: {unknown}")
                samples = [samples_by_id[instance_id] for instance_id in sample_ids]
        elif dataset == "custom":
            if sample_ids is not None:
                raise ValueError(
                    "sample_ids requires dataset='swe-bench-verified'"
                )
            samples = load_samples(
                n_samples,
                sample_id=sample_id,
                samples_path=samples_path,
            )
            if seed_gold_files:
                from eval.harness import _validate_base_commit

                for sample in samples:
                    try:
                        _validate_base_commit(sample.get("base_commit"))
                    except ValueError as exc:
                        raise ValueError(
                            "--seed-gold-files requires every custom sample to "
                            "provide an exact 40-lowercase-hex base_commit"
                        ) from exc
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        results: list[dict[str, Any]] = []

        for i, sample in enumerate(samples):
            print(f"\n{'='*60}", flush=True)
            print(f"Agent v2 sample {i + 1}/{len(samples)}: {sample['id']}", flush=True)
            print(f"{'='*60}", flush=True)
            try:
                result = await evaluate_agent_v2_sample(
                    sample,
                    i,
                    max_retries=max_retries,
                    token_budget=token_budget,
                    seed_gold_files=seed_gold_files,
                )
            except Exception as exc:
                result = safe_failed_sample_result(
                    sample,
                    exc,
                    seed_gold_files=seed_gold_files,
                )
                print(
                    "Warning: agent v2 sample failed; recorded a safe result "
                    "and continued.",
                    file=sys.stderr,
                    flush=True,
                )
            results.append(result)

            path = _write_results_with_fallback(results, results_path)
            print(f"\nAgent v2 eval results saved to {path}", flush=True)
            if predictions_path is not None:
                prediction_file = write_predictions(results, Path(predictions_path))
                print(
                    f"SWE-bench predictions saved to {prediction_file}",
                    flush=True,
                )
        return results
    finally:
        await _close_shared_resources()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python eval/agent_v2_harness.py",
        description="Run RepoPilot's state-graph agent on eval samples.",
    )
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_AGENT_V2_TOKEN_BUDGET,
    )
    parser.add_argument(
        "--seed-gold-files",
        action="store_true",
        help=(
            "Seed relevant_files from custom dataset gold changed files; every "
            "sample must provide an exact 40-lowercase-hex base_commit"
        ),
    )
    parser.add_argument("--samples-file", type=str, default=None,
                        help="Path to custom samples JSONL file")
    parser.add_argument(
        "--dataset",
        choices=("custom", "swe-bench-verified"),
        default="custom",
    )
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--predictions-file", type=str, default=None)
    parser.add_argument("--sample-ids-file", type=str, default=None)
    parser.add_argument("--results-file", type=str, default=None)
    args = parser.parse_args(argv)

    samples_path = Path(args.samples_file) if args.samples_file else SAMPLES_PATH
    sample_ids = None
    if args.sample_ids_file:
        sample_ids = [
            line.strip()
            for line in Path(args.sample_ids_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    asyncio.run(
        run_agent_v2_eval(
            n_samples=args.samples,
            max_retries=args.max_retries,
            token_budget=args.token_budget,
            seed_gold_files=args.seed_gold_files,
            samples_path=samples_path,
            dataset=args.dataset,
            dataset_seed=args.dataset_seed,
            predictions_path=args.predictions_file,
            sample_ids=sample_ids,
            results_path=(
                Path(args.results_file) if args.results_file else RESULTS_PATH
            ),
        )
    )


if __name__ == "__main__":
    main()
