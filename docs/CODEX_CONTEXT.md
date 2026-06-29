# Codex Context Summary

Last updated: 2026-06-11

This file is a local handoff note for Codex sessions. Read it first before searching old chat logs.

## Active Project

- Main repo: `/mnt/e/hermes-work/repopilot`
- GitHub repo: `FMorgan-111/repopilot`
- Related temporary implementation copy: `/tmp/repopilot-work/repopilot-HEAD`
- Related analysis repo: `/tmp/repopilot-analysis`

## Product Direction

RepoPilot is being shaped into a coding/debugging agent for professional programmers.

The user cares more about helping engineers find the real root cause than blindly generating patches. The current design direction is a white-box `plan -> reflect -> execute` agent where the reasoning process is inspectable.

Priority metrics discussed:

1. Final fix rate is the primary metric.
2. Reducing false positives / wrong root-cause judgments is second.
3. Speed to root cause matters, but is lower priority at the current stage.

## Current Architecture Direction

- Use LangGraph as the high-level agent loop.
- Make planning and reflection white-box.
- `DecisionFrame` is intended to become the shared decision protocol across nodes.
- Current near-term value of `DecisionFrame`: align output shape and preserve auditable reasoning.
- Future value of `DecisionFrame`: feed routing, memory, trace replay, and eval.

Core `DecisionFrame` fields discussed:

- `stage`
- `summary`
- `hypotheses`
- `selected_hypothesis_id`
- `evidence`
- `next_checks`
- `recommended_action`
- `risk`
- `confidence`
- `parent_frame_id`
- `trace_notes`

## Last Task Context

The last substantive request before this handoff was:

> "先实现PlanDecision/ReflectDecision"

Meaning:

- Add Pydantic schemas for the full PLAN and REFLECT LLM outputs.
- Do not only validate the embedded `DecisionFrame`.
- Make `plan.py` and `reflect.py` normalize LLM output through `PlanDecision` / `ReflectDecision`.
- Keep backward compatibility with the old flat JSON format.

Implementation status:

- The implementation has been ported into the real repo at `/mnt/e/hermes-work/repopilot`.
- The old temp copy at `/tmp/repopilot-work/repopilot-HEAD` is now only historical reference.
- Added real-repo tests:
  - `tests/test_decision_schemas.py`
  - `tests/test_decision_frame.py`
- Real-repo verification passed:
  - `python3 -m pytest tests/test_decision_schemas.py tests/test_decision_frame.py -q`
  - `python3 -m pytest tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_new_agent.py::test_verify_fix_replans_failed_attempt_once -q`
  - `python3 -m py_compile src/state.py src/schemas.py src/new_agent.py src/nodes/plan.py src/nodes/reflect.py`
  - `ruff check src/state.py src/schemas.py src/new_agent.py src/nodes/plan.py src/nodes/reflect.py tests/test_decision_schemas.py tests/test_decision_frame.py --select=E,F,I --ignore=E501`

Implementation details:

- `src/state.py` now defines `Hypothesis`, `DecisionFrame`, `AgentState.decision_frame`, `AgentState.frame_history`, and `_record_decision_frame`.
- `src/schemas.py` now defines `PlanDecision` and `ReflectDecision`, each validating the embedded `DecisionFrame.stage`.
- `src/nodes/plan.py` normalizes LLM output through `PlanDecision` and records a plan frame.
- `src/nodes/reflect.py` normalizes LLM output through `ReflectDecision` and records a reflect frame.
- `src/new_agent.py` exposes `agent_payload_from_state`, returning `decision_frame` and `frame_history` in the API payload.

## 2026-06-10 Progress

Continued `DecisionFrame` integration in the real repo.

Completed:

- `frame_history` is written into `_save_trace(...)` output, alongside `trace_id`, `steps`, and latest `decision_frame`.
- `agent_payload_from_state(...)` includes `decision_frame` and `frame_history`.
- `DecisionFrame.recommended_action` now participates in routing as a guarded control signal:
  - `route_from_state(...)` first tries to consume the latest valid `DecisionFrame`.
  - It falls back to `current_phase` when there is no frame, no `frame_id`, a stale frame, an already-consumed frame, or an unsupported action.
  - It maps `collect_more_context -> LOCATE`, `plan -> PLAN`, `execute -> EXECUTE`, `reflect -> REFLECT`, `stop -> FAILURE`, and `ask_user -> WAITING_FOR_USER`.
  - If the latest frame recommendation disagrees with `current_phase`, it appends a structured entry to `AgentState.decision_warnings` and logs a warning on `repopilot.graph`.
  - It records each consumed frame once via `decision_route_checked_frame_id` to avoid duplicate routing/warnings.
  - Unknown future actions are still audit-only and fall back with `fallback_reason="unsupported_recommended_action"`.
- Human-input pause support:
  - `Phase.WAITING_FOR_USER` represents a paused run that needs external input.
  - `ask_user` sets `AgentState.pending_human_input=True`, writes `AgentState.human_input_request`, changes `current_phase` to `WAITING_FOR_USER`, and routes to `END`.
  - `human_input_request.question` uses the first nonblank `next_checks[0]`, falling back to the frame summary.
  - `plan_fix(...)` now allows the planner prompt to emit `recommended_action="collect_more_context"` when more repository context is needed, and `recommended_action="ask_user"` when human product decisions, risk authorization, or external facts are required.
  - If `plan_fix(...)` receives a no-patch frame recommending `collect_more_context` or `ask_user`, it preserves `current_phase=PLAN` until routing consumes the frame, so warnings record `actual_phase="PLAN"` rather than a synthetic planner failure.
  - If `plan_fix(...)` receives a no-patch frame that still recommends `execute`, it normalizes the frame to `recommended_action="stop"` and routes to failure instead of executing an empty patch.
  - `agent_payload_from_state(...)` exposes `waiting_for_user` and `human_input_request`.
  - `_save_trace(...)` writes `pending_human_input` and `human_input_request`.
  - `agent_v2` crash payloads also include `waiting_for_user=False` and `human_input_request={}` so callers see a stable shape.
- Route decisions are now captured in `AgentState.route_decisions`, including:
  - `source`: `decision_frame` or `current_phase`
  - `current_phase`
  - `selected_phase`
  - `route`
  - optional `frame_id`, `recommended_action`, and `fallback_reason`
- `_save_trace(...)` and `agent_payload_from_state(...)` include `decision_warnings` and `route_decisions`.
- LangGraph integration detail: conditional edge state mutations are not preserved into final state, so LangGraph nodes record route decisions in `_wrap_node(..., record_route_decision=True)` before the conditional edge runs. The conditional edge then reads the last recorded route via `route_from_recorded_decision(...)`.

Tests added/updated:

- `tests/test_decision_frame.py::test_save_trace_writes_frame_history`
- `tests/test_decision_frame.py::test_plan_fix_no_patch_execute_recommendation_routes_to_failure`
- `tests/test_decision_frame.py::test_plan_fix_collect_more_context_without_patch_routes_to_locate`
- `tests/test_decision_frame.py::test_plan_fix_ask_user_preserves_plan_phase_for_router`
- `tests/test_decision_frame.py::test_save_trace_writes_decision_warnings`
- `tests/test_decision_frame.py::test_route_from_state_records_recommended_action_mismatch_warning`
- `tests/test_decision_frame.py::test_route_from_state_skips_warning_for_aligned_recommended_action`
- `tests/test_decision_frame.py::test_route_from_state_consumes_each_decision_frame_once`
- `tests/test_decision_frame.py::test_route_from_state_consumes_supported_recommended_actions`
- `tests/test_decision_frame.py::test_route_from_state_consumes_collect_more_context_recommendation`
- `tests/test_decision_frame.py::test_route_from_state_consumes_ask_user_as_human_input_pause`
- `tests/test_decision_frame.py::test_route_from_state_uses_summary_as_human_input_question_when_no_next_checks`
- `tests/test_decision_frame.py::test_route_from_state_uses_summary_as_human_input_question_when_first_check_blank`
- `tests/test_decision_frame.py::test_route_from_state_falls_back_for_unsupported_recommended_action`
- `tests/test_decision_frame.py::test_route_from_state_falls_back_when_decision_frame_has_no_id`
- `tests/test_decision_frame.py::test_route_from_state_falls_back_for_stale_decision_frame`
- `tests/test_decision_frame.py::test_save_trace_writes_route_decisions`
- `tests/test_decision_frame.py::test_save_trace_writes_human_input_request`
- `tests/test_decision_frame.py::test_agent_payload_exposes_route_decisions`
- `tests/test_decision_frame.py::test_agent_payload_exposes_human_input_pause`
- `tests/test_new_agent.py::test_agent_v2_crash_payload_exposes_human_input_defaults`
- `tests/test_new_agent.py::test_agent_v2_state_machine_transitions_to_done` now asserts final LangGraph state includes persisted route decisions.
- `tests/test_new_agent.py::test_langgraph_conditional_router_uses_native_async_callable`

LangGraph hang fix:

- `tests/test_new_agent.py::test_agent_v2_state_machine_transitions_to_done` previously hung in the installed LangGraph async branch after `locate_code -> PLAN`.
- Root cause: `StateGraph.add_conditional_edges(..., route_from_state, ...)` registered the synchronous router. In LangGraph `ainvoke()`, that sync path is wrapped as `run_in_executor`, and local minimal repros hung after the first conditional edge.
- Fix: `build_agent_graph()` now registers an async conditional edge function for LangGraph. The fallback graph still calls synchronous `route_from_state(...)` directly.
- Regression guard: `test_langgraph_conditional_router_uses_native_async_callable` checks that the compiled LangGraph branch router is not the executor wrapper.

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_tracer.py -q` -> 41 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/state.py src/schemas.py src/new_agent.py src/graph.py src/nodes/plan.py src/nodes/reflect.py src/tracer.py`
- `ruff check src/state.py src/schemas.py src/new_agent.py src/graph.py src/nodes/plan.py src/nodes/reflect.py src/tracer.py tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_tracer.py tests/test_new_agent.py --select=E,F,I --ignore=E501`

## Repo State Notes

`/mnt/e/hermes-work/repopilot` currently appears to be the real long-term repo.

Known local status there during recovery:

- Modified: `eval/eval_results.json`
- Modified by this task: `src/state.py`, `src/schemas.py`, `src/new_agent.py`, `src/graph.py`, `src/nodes/plan.py`, `src/nodes/reflect.py`, `src/tracer.py`
- Added by this task: `docs/CODEX_CONTEXT.md`, `tests/test_decision_schemas.py`, `tests/test_decision_frame.py`
- Untracked: `logs/`, `scripts/`, `shards/`, `test_tox_single.py`

Do not overwrite or delete these unless the user explicitly asks.

`/tmp/repopilot-analysis` is a git repo with a different set of white-box changes:

- Structured `patch_edits`
- Hypothesis fields in state
- Plan/reflection white-box prompt changes

Be careful not to blindly merge `/tmp/repopilot-analysis` into `/mnt/e/hermes-work/repopilot`; compare intent first.

## 2026-06-11 Progress

Implemented the minimum durable-pause and resume slice in the real repo.

Completed:

- Added `src/run_store.py`.
  - `default_runs_dir()` returns `~/.repopilot`.
  - `run_path(run_id, root_dir=None)` maps a run id to `runs/{run_id}.json`.
  - `save_run(state, root_dir=None)` writes `AgentState.model_dump(mode="json")`.
  - `load_run(run_id, root_dir=None)` reloads and validates an `AgentState`.
- `agent_payload_from_state(...)` now includes `run_id`, currently equal to `trace_id`.
- `agent_v2(...)` saves paused runs when final phase is `WAITING_FOR_USER`.
- Added `resume_agent_v2(run_id, human_answer)`.
  - Loads a saved paused run.
  - Rejects non-paused runs with a stable error payload.
  - Appends the human answer to `conversation_history`.
  - Clears `pending_human_input` and `human_input_request`.
  - Restarts from `Phase.PLAN`.
  - Preserves/sets consumed-frame semantics through `decision_route_checked_frame_id`, so the old `ask_user` frame falls back instead of re-pausing.
  - Saves the run again if the resumed graph pauses again.
- `build_agent_graph(start_phase=Phase.UNDERSTAND)` now accepts a start phase.
  - Normal `agent_v2(...)` still starts from `UNDERSTAND`.
  - `resume_agent_v2(...)` starts from `PLAN`.

Important design clarification:

- On resume, `already_consumed` fallback does not move the agent by itself.
- Resume explicitly sets `current_phase=PLAN`.
- The old frame then falls back to `current_phase`, so routing enters `plan_fix`.

Tests added/updated:

- `tests/test_run_store.py::test_save_and_load_paused_run_preserves_pause_state`
- `tests/test_new_agent.py::test_agent_v2_saves_waiting_for_user_run`
- `tests/test_new_agent.py::test_agent_v2_starts_graph_at_understand`
- `tests/test_new_agent.py::test_resume_agent_v2_rejects_non_paused_run`
- `tests/test_new_agent.py::test_resume_agent_v2_injects_answer_and_resumes_from_plan`
- `tests/test_new_agent.py::test_resume_agent_v2_starts_graph_at_plan`
- `tests/test_new_agent.py::test_resume_agent_v2_saves_run_when_it_pauses_again`

Fresh verification passed:

- `python3 -m pytest tests/test_run_store.py tests/test_new_agent.py tests/test_decision_frame.py -q` -> 36 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/new_agent.py src/run_store.py src/state.py src/graph.py`
- `ruff check src/new_agent.py src/run_store.py tests/test_new_agent.py tests/test_run_store.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`

## 2026-06-11 FastAPI Resume Interface

Exposed resume through the FastAPI layer in the real repo.

Completed:

- Added `AgentV2ResumeRequest` in `src/main.py` with `run_id` and `human_answer`.
- Added `POST /agent/v2/resume`.
- The endpoint calls `resume_agent_v2(run_id, human_answer)` and returns the agent payload directly on success.
- Resume client errors now map to HTTP 400:
  - existing `"Invalid ..."` errors
  - `"Run {run_id} is not waiting for user input."`
- Other agent errors still map to HTTP 502.
- `tests/test_main.py` now uses `httpx.AsyncClient` with `ASGITransport` instead of `fastapi.testclient.TestClient`.
  - Reason: in this local Python/FastAPI/Starlette stack, `TestClient.get()` hangs even for a minimal FastAPI app, while `ASGITransport` returns correctly.

Tests added/updated:

- `tests/test_main.py::test_post_agent_v2_resume_routes_to_resume_agent`
- `tests/test_main.py::test_post_agent_v2_resume_rejects_non_paused_run`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_main.py -q` -> 8 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_main.py tests/test_new_agent.py tests/test_run_store.py -q` -> 19 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/main.py tests/test_main.py src/new_agent.py src/run_store.py`
- `ruff check src/main.py tests/test_main.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`

## 2026-06-11 Resume Interface Follow-ups

Used a subagent for the API error-contract sidecar while the main session implemented CLI resume.

Completed:

- API resume now catches saved-run load failures at the FastAPI boundary:
  - missing run file -> HTTP 404 with `{"status": "error", "success": false, "run_id": ..., "error": "Saved run ... was not found."}`
  - corrupt JSON or invalid saved `AgentState` -> HTTP 500 with `{"status": "error", "success": false, "run_id": ..., "error": "Saved run ... could not be loaded."}`
  - non-paused run remains HTTP 400
- Added CLI resume:
  - `repopilot resume <run_id> <human_answer>`
  - `--json` prints the stable payload shape
  - legacy `repopilot <issue_url>` still works
- `src/cli.py` now accepts `main(argv=None)` for testability.

Tests added/updated:

- `tests/test_main.py::test_post_agent_v2_resume_returns_404_for_missing_saved_run`
- `tests/test_main.py::test_post_agent_v2_resume_returns_500_for_corrupt_saved_run_json`
- `tests/test_main.py::test_post_agent_v2_resume_returns_500_for_invalid_saved_run_state`
- `tests/test_cli.py::test_cli_resume_json_calls_resume_agent`
- `tests/test_cli.py::test_cli_issue_url_path_remains_supported`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_main.py -q` -> 11 passed
- `python3 -m pytest -p no:cacheprovider tests/test_cli.py -q` -> 2 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_cli.py tests/test_main.py tests/test_new_agent.py tests/test_run_store.py -q` -> 24 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/main.py tests/test_main.py src/cli.py tests/test_cli.py`
- `ruff check src/main.py tests/test_main.py src/cli.py tests/test_cli.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`

## 2026-06-11 Saved Run Listing/Inspect

Implemented the read-only saved-run inspection slice.

Completed:

- Added run summary helpers in `src/run_store.py`:
  - `runs_dir(root_dir=None)`
  - `inspect_run(run_id, root_dir=None)`
  - `list_runs(root_dir=None)`
  - `summarize_run(state, path=None)`
- Summary shape includes:
  - `run_id`
  - `issue_url`
  - `current_phase`
  - `pending_human_input`
  - `human_input_question`
  - `latest_decision_frame`
  - `updated_at`
- Added CLI commands:
  - `repopilot runs`
  - `repopilot runs --json`
  - `repopilot inspect <run_id>`
  - `repopilot inspect <run_id> --json`

Tests added/updated:

- `tests/test_run_store.py::test_inspect_run_returns_stable_summary`
- `tests/test_run_store.py::test_list_runs_returns_saved_run_summaries_sorted_by_run_id`
- `tests/test_cli.py::test_cli_runs_json_lists_saved_runs`
- `tests/test_cli.py::test_cli_inspect_json_returns_saved_run_summary`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_run_store.py -q` -> 3 passed
- `python3 -m pytest -p no:cacheprovider tests/test_cli.py -q` -> 4 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_cli.py tests/test_run_store.py tests/test_main.py tests/test_new_agent.py -q` -> 28 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/run_store.py src/cli.py tests/test_run_store.py tests/test_cli.py`
- `ruff check src/run_store.py src/cli.py tests/test_run_store.py tests/test_cli.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`
- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_明日待办.md`

## 2026-06-11 Trace Replay Summary

Implemented the read-only trace replay summary slice for saved runs.

Completed:

- Added replay helpers in `src/run_store.py`:
  - `replay_run(run_id, root_dir=None)`
  - `summarize_replay(state)`
- Replay output includes:
  - `run_id`
  - `issue_url`
  - `current_phase`
  - `pause.pending_human_input`
  - `pause.question`
  - `pause.request`
  - chronological `timeline`
- Timeline entries include decision frames with:
  - frame id/stage/summary
  - selected hypothesis id and selected hypothesis details when present
  - recommended action
  - risk/confidence
  - matched route decision by `frame_id`
  - matched warnings by `frame_id`
  - next checks and trace notes
- Route decisions without a matching `frame_id` are preserved as standalone timeline entries.
- Added CLI command:
  - `repopilot replay <run_id>`
  - `repopilot replay <run_id> --json`

Tests added/updated:

- `tests/test_run_store.py::test_replay_run_returns_white_box_timeline`
- `tests/test_cli.py::test_cli_replay_json_returns_trace_replay`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_run_store.py::test_replay_run_returns_white_box_timeline -q`
- `python3 -m pytest -p no:cacheprovider tests/test_cli.py::test_cli_replay_json_returns_trace_replay -q`
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_cli.py tests/test_run_store.py tests/test_main.py tests/test_new_agent.py -q` -> 30 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/run_store.py src/cli.py tests/test_run_store.py tests/test_cli.py`
- `ruff check src/run_store.py src/cli.py tests/test_run_store.py tests/test_cli.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`
- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_明日待办.md`

## 2026-06-11 Replay Markdown Formatter

Implemented the Markdown formatter for replay summaries.

Completed:

- Added `format_replay_markdown(replay)` in `src/run_store.py`.
- Markdown output includes:
  - title with run id
  - issue URL
  - final phase
  - pending human input flag
  - pause question when present
  - timeline sections for decision frames
  - selected hypothesis id and claim
  - recommended action, risk, confidence
  - actual route
  - route/frame warning summary
  - next checks and trace notes
  - standalone route-decision sections
- Added CLI flag:
  - `repopilot replay <run_id> --markdown`

Tests added/updated:

- `tests/test_run_store.py::test_format_replay_markdown_summarizes_timeline`
- `tests/test_cli.py::test_cli_replay_markdown_prints_formatted_replay`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_run_store.py::test_format_replay_markdown_summarizes_timeline -q`
- `python3 -m pytest -p no:cacheprovider tests/test_cli.py::test_cli_replay_markdown_prints_formatted_replay -q`
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_cli.py tests/test_run_store.py tests/test_main.py tests/test_new_agent.py -q` -> 32 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/run_store.py src/cli.py tests/test_run_store.py tests/test_cli.py`
- `ruff check src/run_store.py src/cli.py tests/test_run_store.py tests/test_cli.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`
- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_明日待办.md`

## 2026-06-11 FastAPI Replay Interface

Exposed replay summaries through the FastAPI layer.

Completed:

- Added `GET /agent/v2/runs/{run_id}/replay` in `src/main.py`.
- Default response returns the JSON replay summary from `replay_run(run_id)`.
- `?format=markdown` returns `format_replay_markdown(replay)` with `text/markdown`.
- Saved-run load errors match the existing resume contract:
  - missing run -> HTTP 404 with stable error payload
  - corrupt JSON or invalid saved `AgentState` -> HTTP 500 with stable error payload

Tests added/updated:

- `tests/test_main.py::test_get_agent_v2_replay_returns_json`
- `tests/test_main.py::test_get_agent_v2_replay_returns_markdown`
- `tests/test_main.py::test_get_agent_v2_replay_returns_404_for_missing_saved_run`
- `tests/test_main.py::test_get_agent_v2_replay_returns_500_for_corrupt_saved_run`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_main.py -q` -> 15 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_main.py tests/test_cli.py tests/test_run_store.py tests/test_new_agent.py -q` -> 36 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/main.py tests/test_main.py`
- `ruff check src/main.py tests/test_main.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Desktop explanation updated:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_代理层与路由层详解.md`
- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_明日待办.md`

## 2026-06-11 Agent V2 Eval Replay Workflow

Implemented the full agent-v2 eval/replay slice, not the smaller report-only slice.

Completed:

- Added optional final-run persistence to `agent_v2(...)`:
  - new parameter `save_final_run=False`
  - normal API/CLI behavior unchanged by default
  - eval can call `agent_v2(..., save_final_run=True)` so DONE/FAILED/WAITING_FOR_USER states are saved under `~/.repopilot/runs/{trace_id}.json`
- Added `eval/agent_v2_harness.py`.
  - loads dataset samples
  - calls the real state-graph `agent_v2`
  - writes `mode="agent_v2"` result rows to `eval/eval_results.json`
  - stores `run_id`, `trace_id`, final phase, success/waiting flags, turns, token usage, original agent payload, replay summary, and replay load errors
- Added `python3 eval/harness.py --agent-v2`.
  - default `python3 eval/harness.py` still runs the legacy two-phase eval
  - `--agent-v2 --samples N --max-retries N --token-budget N` dispatches to the new runner
- Updated `eval/report.py`.
  - legacy rows keep the existing file_recall/patch_apply/test_pass metrics
  - agent-v2 rows add aggregate `agent_v2_samples`, `agent_v2_success_rate`, and `agent_v2_waiting_for_user`
  - `mode="agent_v2"` rows render an `Agent V2 Results` table
  - replay summaries render a `Replay Diagnostics` section with latest decision frame, selected hypothesis, claim, recommended action, actual route, warnings, and next checks
- Fixed direct script execution for eval scripts by bootstrapping repo root into `sys.path`.
- Wrote implementation plan:
  - `docs/superpowers/plans/2026-06-11-agent-v2-eval-replay.md`

Tests added/updated:

- `tests/test_new_agent.py::test_agent_v2_saves_final_run_when_requested`
- `tests/test_agent_v2_eval.py`
- `tests/test_eval_report.py`

Fresh verification passed:

- `python3 -m pytest -p no:cacheprovider tests/test_new_agent.py::test_agent_v2_saves_final_run_when_requested -q`
- `python3 -m pytest -p no:cacheprovider tests/test_agent_v2_eval.py -q` -> 3 passed
- `python3 -m pytest -p no:cacheprovider tests/test_eval_report.py -q` -> 1 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m pytest -p no:cacheprovider tests/test_agent_v2_eval.py tests/test_eval_report.py tests/test_new_agent.py tests/test_run_store.py -q` -> 20 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/new_agent.py eval/agent_v2_harness.py eval/harness.py eval/report.py tests/test_agent_v2_eval.py tests/test_eval_report.py`
- `ruff check src/new_agent.py eval/agent_v2_harness.py eval/harness.py eval/report.py tests/test_agent_v2_eval.py tests/test_eval_report.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`
- `python3 eval/harness.py --help`
- `python3 eval/agent_v2_harness.py --help`

Preset workflow now available:

1. `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 10000`
2. `python3 eval/report.py`
3. Inspect `eval/eval_summary.md`, especially `Agent V2 Results` and `Replay Diagnostics`
4. Use `repopilot replay <run_id> --markdown` for deeper one-run analysis

Not run in this session:

- A live `--agent-v2` eval sample, because it would make real network/LLM/GitHub calls and may take time/cost money.

## Next Recommended Step

## 2026-06-11 Agent V2 Diagnostics And Frame Health

Followed up on the live one-sample agent-v2 eval failure.

Root-cause finding:

- Re-running `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000` still failed.
- The saved result had `final_phase=FAILED`, `error="Phase plan_fix timed out after 90.0s"`, and `token_used=5595`.
- Therefore the observed failure was not caused by the agent token budget being too low.
- The immediate problem was phase timeout alignment: `plan_fix`/`reflect_on_failure` were shorter than the LLM request timeout * retry window.

Completed:

- Added `AgentState.node_diagnostics`.
- Added `_record_node_diagnostic(...)` and `_describe_exception(...)` in `src/state.py`.
- `plan_fix(...)` and `reflect_on_failure(...)` now record LLM diagnostics:
  - `node`
  - `event`
  - `status`
  - `elapsed_seconds`
  - `error_type` / `error` on failure
  - `prompt_tokens_estimate`
  - `response_tokens_estimate` on success
- Empty exception messages, especially `asyncio.TimeoutError()`, now produce useful text such as `TimeoutError` instead of `Failed to generate fix plan: `.
- LangGraph node wrapper and fallback graph timeout/error paths now record phase diagnostics, so outer `asyncio.wait_for(...)` cancellations are visible.
- `agent_payload_from_state(...)`, crash payloads, `_save_trace(...)`, saved-run replay JSON, and replay Markdown now expose node diagnostics.
- Added frame-health audit warnings for legacy flat LLM output that omits explicit `decision_frame`:
  - warning entries use `warning_type="frame_health"`
  - reason currently `missing_explicit_decision_frame`
  - backward compatibility remains: legacy output still normalizes into `PlanDecision` / `ReflectDecision`
- Raised `PHASE_TIMEOUTS["plan_fix"]` and `PHASE_TIMEOUTS["reflect_on_failure"]` to `150.0`.
- Follow-up: after a live eval timed out in `understand_issue`, raised `PHASE_TIMEOUTS["understand_issue"]` to `150.0` and extended `test_llm_phase_timeouts_cover_retry_window` to cover it.
- Added implementation plan:
  - `docs/superpowers/plans/2026-06-11-agent-v2-diagnostics.md`

Tests added/updated:

- `tests/test_decision_frame.py::test_plan_fix_records_llm_diagnostic_on_timeout`
- `tests/test_decision_frame.py::test_plan_fix_records_successful_llm_diagnostic`
- `tests/test_decision_frame.py::test_agent_payload_exposes_node_diagnostics`
- `tests/test_decision_frame.py::test_plan_fix_records_frame_health_warning_for_legacy_output`
- `tests/test_decision_frame.py::test_reflect_records_frame_health_warning_for_legacy_output`
- `tests/test_decision_frame.py::test_save_trace_writes_node_diagnostics`
- `tests/test_run_store.py::test_replay_run_includes_node_diagnostics`
- `tests/test_run_store.py::test_format_replay_markdown_includes_node_diagnostics`
- `tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window`
- `tests/test_new_agent.py::test_fallback_graph_records_phase_timeout_diagnostic`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_decision_frame.py tests/test_run_store.py tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window tests/test_new_agent.py::test_fallback_graph_records_phase_timeout_diagnostic -q` -> 40 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_main.py tests/test_cli.py tests/test_agent_v2_eval.py tests/test_eval_report.py -q` -> 25 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/state.py src/new_agent.py src/run_store.py src/nodes/plan.py src/nodes/reflect.py src/graph.py tests/test_decision_frame.py tests/test_run_store.py tests/test_new_agent.py`
- `ruff check src/state.py src/new_agent.py src/run_store.py src/nodes/plan.py src/nodes/reflect.py src/graph.py tests/test_decision_frame.py tests/test_run_store.py tests/test_new_agent.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step after this slice:

- Re-run a live one-sample agent-v2 eval with the new diagnostics and inspect the replay Markdown. If `plan_fix` still times out at 150s, the replay should now show whether the failure came from phase timeout, LLM request timeout, HTTP status, or malformed/slow LLM output.

The next useful slice is either validating the new eval path with a live small batch or adding report archival. Keep it separate from memory changes unless explicitly requested:

1. Run a real one-sample agent-v2 eval with API credentials and inspect `Replay Diagnostics`.
2. Add optional report archival, e.g. save per-run replay markdown under an eval reports directory.
3. Consider a dedicated `eval/report.py --agent-v2-only` or `--markdown` flag if the mixed legacy/agent-v2 report gets noisy.

Do not combine eval harness integration and report archival in one slice unless the user asks.

Historical note: durable paused-state persistence, core resume, FastAPI resume exposure, API load-error contracts, CLI resume, saved-run listing/inspect, CLI trace replay, replay Markdown formatting, FastAPI replay exposure, and agent-v2 eval replay reporting are now implemented as minimum slices.

## Previous Recommended Step

The original first engineering step was to continue from the new human-input pause surface with the smallest durable-pause slice. Do **not** start with full resume behavior.

Minimum design for step 1:

1. Add a focused persistence module, likely `src/run_store.py`.
2. Store paused run state under `~/.repopilot/runs/{trace_id}.json`.
3. Use `trace_id` as the first `run_id`.
4. Save the complete `AgentState.model_dump(mode="json")` when `agent_v2(...)` ends in `Phase.WAITING_FOR_USER`.
5. Add `run_id` to `agent_payload_from_state(...)` so callers can later resume or inspect the paused run.
6. Add tests that save and reload a paused `AgentState`, preserving `human_input_request`, `frame_history`, and `route_decisions`.

Keep this separate from resume:

- Resume is more complex because it must load a run, inject a human answer, clear pause flags, pick the correct restart phase (`PLAN`, `LOCATE`, or `REFLECT`), avoid re-consuming stale `DecisionFrame`s, and continue the graph.
- Implement resume only after durable paused-state persistence is verified.

Subagent decision:

- For the minimum persistence slice, prefer a single agent/session. The change is small and touches shared files (`src/run_store.py`, `src/new_agent.py`, tests), so splitting would likely add coordination overhead.
- Consider subagents for the later resume work, where persistence, CLI/API entrypoint, state-transition rules, and trace replay can be separated.

Rough estimate discussed:

- Single-agent minimum persistence: about 1.5-2.5 hours.
- Two-subagent version: about 1-1.5 hours plus integration/review, with limited net savings.

Original broader roadmap remains:

1. Durable `AgentState` persistence for paused `WAITING_FOR_USER` runs.
2. Resume entrypoint accepting a human answer and continuing from `PLAN`, `LOCATE`, or `REFLECT`.
3. Eval/trace replay tooling that summarizes `frame_history`, `decision_warnings`, `route_decisions`, and `human_input_request` for a saved run.

The earlier broad plan was copied into the desktop tomorrow TODO:

- `/mnt/c/Users/admin/Desktop/RepoPilot/RepoPilot_明日待办.md`

## 2026-06-11 Live Agent V2 Eval Validation

Ran the recommended live one-sample agent-v2 eval after diagnostics/frame-health work.

Command run outside the sandbox because the eval needs network access and writes `~/.repopilot` plus eval artifacts:

- `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`

Result:

- Sample: `tox-dev/tox#3075:3748`
- Run/trace id: `278aad07a2b0`
- Final phase: `FAILED`
- Error: `Maximum retries reached: 1.`
- Turns: `26`
- Token used: `22579`
- Eval artifacts updated:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`
- Saved run replay is available through `replay_run("278aad07a2b0")` and `repopilot replay 278aad07a2b0 --markdown` in a normal writable environment.

Key diagnosis from replay Markdown:

- The diagnostics changes worked: no `plan_fix`, `reflect_on_failure`, or `understand_issue` phase timeout occurred in this run.
- Node diagnostics captured four successful LLM calls:
  - `plan_fix` success, ~47.476s, prompt estimate 2286, response estimate 790
  - `plan_fix` success, ~58.807s, prompt estimate 2286, response estimate 1404
  - `reflect_on_failure` success, ~52.46s, prompt estimate 1031, response estimate 685
  - `plan_fix` success, ~102.076s, prompt estimate 2992, response estimate 1051
- The agent failed for agent-quality/retry reasons, not infrastructure timeout:
  - first plan requested `collect_more_context`; router returned to `locate_code` and recorded the expected warning
  - second plan generated a patch, but patch application failed due to malformed unified diff
  - reflection identified malformed diff formatting
  - final plan produced another patch but verification failed and max retries was reached
- Replay showed the later hypothesis drifted from envpython substitution to env uniqueness hashing, so the next agent-quality slice should focus on patch generation/apply validation and hypothesis consistency after reflection.

Additional operational finding:

- In the Codex sandbox, the first eval attempt failed before LLM planning because `src/cache.py` tried to write `~/.repopilot/cache`, and saving final runs tried to write `~/.repopilot/runs`. This is expected in the restricted sandbox but worth making configurable or gracefully degraded for sandboxed/dev runs.
- The live eval process printed `Agent v2 eval results saved` but did not exit cleanly until the exact PID was killed. This looks like an async/client/thread cleanup issue in the eval path. Treat it as a separate harness cleanup bug if it repeats.

Next recommended step after this validation:

1. Add a focused patch-quality/retry slice: ensure generated patches are apply-checked before consuming a retry, and preserve/refine the selected hypothesis after reflection instead of allowing unrelated root-cause drift.
2. Separately, add eval harness cleanup so live eval exits after writing results, likely by closing async HTTP clients/executors used by the graph/tools.
3. Optionally add configurable `REPOPILOT_HOME`/run-store/cache root for sandboxed local runs.

## 2026-06-11 Patch Apply Failure Retry Slice

Implemented task 1 from the live eval follow-up: malformed/unapplyable unified diffs no longer immediately consume the normal debug retry.

Completed:

- Added `FixAttempt.failure_kind` in `src/state.py`.
- `src/nodes/execute.py` now tags failure kinds:
  - patch apply failure -> `patch_apply_failed`
  - test failure -> `test_failed`
  - execution exception -> `execution_error`
- `src/nodes/verify.py` now handles patch-apply failures separately:
  - first consecutive patch apply failure routes to `REFLECT` without incrementing `retry_count`
  - second or later consecutive patch apply failure consumes normal retry budget
  - identical patch + identical error still uses the existing same-failure guard and routes to `FAILURE`
  - legacy saved attempts with `test_result="patch_apply_failed"` and empty `failure_kind` are still recognized
- `src/nodes/reflect.py` now detects patch apply failures via either `failure_kind` or legacy `test_result`.
- Reflection prompt for patch apply failures now explicitly says:
  - tests never ran because the unified diff failed to apply
  - keep the selected root-cause hypothesis unless apply output proves the target file/context is impossible
  - focus the next action on unified diff formatting, file path, and hunk context repair
  - include concise previous selected-hypothesis context from `decision_frame` / `frame_history`

Subagents used:

- Worker Rawls implemented `state` / `execute` / `verify` retry semantics.
- Worker Beauvoir implemented reflection prompt guard and tests.
- Reviewer Bohr found missing direct tests for `execute_fix` failure tagging; fixed by adding direct execute tests.
- Reviewer Noether found `patch_apply_failed` bypassed `_same_failure_seen_twice`; fixed by moving same-failure guard before the patch-apply special case and adding a regression test.

Tests added/updated:

- `tests/test_new_agent.py::test_execute_fix_marks_patch_apply_failure_kind`
- `tests/test_new_agent.py::test_execute_fix_marks_test_failure_kind`
- `tests/test_new_agent.py::test_execute_fix_marks_execution_error_failure_kind`
- `tests/test_new_agent.py::test_verify_fix_first_patch_apply_failure_does_not_increment_retry_count`
- `tests/test_new_agent.py::test_verify_fix_legacy_patch_apply_failure_does_not_increment_retry_count`
- `tests/test_new_agent.py::test_verify_fix_second_consecutive_patch_apply_failure_consumes_retry`
- `tests/test_new_agent.py::test_verify_fix_same_patch_apply_failure_twice_routes_to_failure`
- `tests/test_new_agent.py::test_verify_fix_test_failure_still_increments_retry_count`
- `tests/test_decision_frame.py::test_reflect_on_patch_apply_failure_preserves_selected_hypothesis_context`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_frame.py -q` -> 55 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/state.py src/nodes/execute.py src/nodes/verify.py src/nodes/reflect.py tests/test_new_agent.py tests/test_decision_frame.py`
- `ruff check src/state.py src/nodes/execute.py src/nodes/verify.py src/nodes/reflect.py tests/test_new_agent.py tests/test_decision_frame.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Implement task 2 from the live eval follow-up: preserve/refine the selected root-cause hypothesis across reflection and the next planning pass, so a patch-format failure does not cause root-cause drift.
2. Then rerun the same one-sample live agent-v2 eval (`tox-dev/tox#3075:3748`) and compare replay against run `278aad07a2b0`.

## 2026-06-11 Effect-Level MVP Gap And BM25 Reranking Todo

Current status:

- Engineering-level MVP is in place: LangGraph agent loop, `DecisionFrame`, plan/reflect schemas, pause/resume, CLI/API, saved-run replay, eval harness, and diagnostics.
- Effect-level MVP is not complete yet. The minimum remaining proof is one real GitHub issue that reaches an apply-able patch and a defensible verification result, with replay explaining the decision chain.

Effect-level MVP remaining work:

1. Hypothesis consistency after reflection:
   - Live run `278aad07a2b0` drifted from the selected `envpython` hypothesis to an unrelated environment-hash hypothesis after reflection.
   - Next implementation slice should preserve/refine `selected_hypothesis_id` across `REFLECT -> PLAN`.
   - For patch-format failures, the planner should treat reflection as patch repair guidance, not as evidence to change root cause.
2. Rerun live eval:
   - Command: `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
   - Compare replay against run `278aad07a2b0`.
3. Small-batch eval:
   - Run 3-5 samples after the hypothesis-consistency slice.
   - Report categories: file-location failure, root-cause failure, patch-apply failure, test/verification failure, infra/sandbox failure.
4. Demo report:
   - Produce one markdown report showing `plan -> reflect -> execute` decisions and why the final result succeeded or failed.

BM25 reranking is a tomorrow task, but not the top blocker for effect-level MVP.

BM25 decision:

- Add BM25 to RepoPilot as a deterministic lexical reranker, not as a full RAG/vector-search system.
- Do not claim RepoPilot uses FAISS unless a real optional FAISS path is implemented later.
- Keep FAISS in the SciSift project for resume claims.
- Resume wording should be `GitHub Code Search + BM25 reranking`, not `BM25/FAISS RAG`.

Recommended BM25 implementation boundary:

1. Create `src/retrieval.py`.
2. Implement pure-Python BM25 to avoid adding dependency risk:
   - tokenize issue query, file path, and file content
   - compute BM25 over the hydrated candidate files returned by existing GitHub Code Search
   - normalize score to blend into `FileInfo.relevance_score`
3. Integrate in `src/nodes/locate.py` after candidate files are read:
   - current flow: GitHub Code Search -> rank by path heuristic -> read top files
   - tomorrow flow: GitHub Code Search -> rank/read candidate files -> BM25 rerank hydrated files -> pass top files to planner
4. Record trace/tool call:
   - tool name: `bm25_rerank`
   - args: query text, candidate count
   - result: ranked paths and scores
5. Update `FileInfo.reason` with BM25 evidence, e.g. `bm25 rerank score=...; matched issue terms`.
6. Tests:
   - BM25 ranks a file containing exact issue terms above an unrelated file.
   - `locate_code` records `bm25_rerank`.
   - `locate_code` preserves existing behavior if file content is empty or BM25 has no useful matches.

Important technical decisions from today:

- Patch apply failure is not a semantic code failure; it is a patch-quality failure.
- First patch apply failure gets one free reflection without incrementing normal `retry_count`.
- Repeated identical malformed patch + identical error still triggers same-failure guard and routes to `FAILURE`.
- Reflection prompt now explicitly says tests never ran and should preserve the selected root-cause hypothesis unless the apply error disproves the file/context.
- Subagent review caught two useful gaps:
  - missing direct `execute_fix` tests for `failure_kind`
  - same-failure guard was initially bypassed for patch apply failures

Operational blockers observed today:

- Restricted Codex sandbox cannot write `~/.repopilot/cache` or `~/.repopilot/runs`; live eval requires escalation or configurable `REPOPILOT_HOME`.
- Live eval process printed saved results but did not exit cleanly; likely async HTTP/client/executor cleanup. Treat as separate eval harness cleanup task.

## 2026-06-12 Hypothesis Consistency After Patch Apply Failure

Implemented task 2 from the live eval follow-up: prevent root-cause drift from `envpython` to unrelated hypotheses after a malformed patch fails to apply.

Completed:

- `src/nodes/plan.py` now detects when the latest fix attempt failed with `patch_apply_failed`.
- For patch-apply failures, planner prompts include a structured `Hypothesis Continuity Instructions` block:
  - tests did not run, so reflection is patch-repair guidance rather than root-cause disproof
  - preserve the selected hypothesis from the previous plan frame unless the apply error proves the file/hunk target impossible
  - include the previous selected hypothesis id, claim, evidence, and latest reflection summary
- `plan_fix(...)` now applies a defensive normalization after LLM output for patch-apply failures:
  - if the next plan frame selects a different hypothesis, the selected hypothesis is restored to the previous plan-frame anchor
  - the anchored hypothesis is inserted into the new frame if missing
  - a `decision_warnings` entry with `warning_type="hypothesis_consistency"` and reason `preserved_selected_hypothesis_after_patch_apply_failure` records the LLM-selected id
- Normal test failures and semantic verification failures are unchanged; the guard only applies after `patch_apply_failed`.

Tests added:

- `tests/test_decision_frame.py::test_plan_fix_after_patch_apply_failure_includes_hypothesis_anchor`
- `tests/test_decision_frame.py::test_plan_fix_after_patch_apply_failure_restores_drifted_hypothesis`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_decision_frame.py::test_plan_fix_after_patch_apply_failure_includes_hypothesis_anchor tests/test_decision_frame.py::test_plan_fix_after_patch_apply_failure_restores_drifted_hypothesis -q`
- `python3 -B -m pytest -p no:cacheprovider tests/test_decision_frame.py tests/test_new_agent.py -q` -> 57 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_run_store.py tests/test_eval_report.py -q` -> 9 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/nodes/plan.py tests/test_decision_frame.py`
- `ruff check src/nodes/plan.py tests/test_decision_frame.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Re-run the same one-sample live agent-v2 eval (`tox-dev/tox#3075:3748`):
   - `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
2. Compare replay against run `278aad07a2b0`, specifically checking whether the post-reflection plan preserves the `envpython` selected hypothesis and whether any remaining failure is patch syntax, verification, or root-cause quality.
3. Separately address the eval process cleanup if it still hangs after writing results.

## 2026-06-12 Live Eval Rerun After Hypothesis Guard

Reran the same one-sample live agent-v2 eval:

- Command: `timeout 900s python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Sample: `tox-dev/tox#3075:3748`
- Run/trace id: `4b44e8fe0149`
- Final phase: `FAILED`
- Error: `Maximum retries reached: 1.`
- Token used: `14720`
- Report regenerated with `python3 eval/report.py`.
- Updated artifacts:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`
  - saved run: `~/.repopilot/runs/4b44e8fe0149.json`

Diagnosis:

- This run did **not** exercise the patch-apply failure path fixed above.
- Both fix attempts failed before patch validation because the execution node could not clone `tox-dev/tox`:
  - first attempt: direct GitHub clone failed to connect to port 443
  - second attempt: tokenized GitHub clone timed out after 180 seconds
- `FixAttempt.failure_kind` for both attempts was `execution_error`, not `patch_apply_failed`.
- `decision_warnings` was empty, so the new `hypothesis_consistency` guard did not trigger.
- Replay frames:
  - `df_0001` selected an id/environment identity hypothesis in `src/tox/tox_env/runner.py`
  - `df_0002` reflection treated the failure as network/infrastructure and suggested checking `envpython`/cache layers
  - `df_0003` selected another id/environment identity hypothesis in `src/tox/tox_env/api.py`
- The run is therefore inconclusive for the original malformed-diff/envpython drift fix; it primarily validates that eval execution is currently blocked by clone reliability.

Operational findings:

- The eval process again printed saved results but did not exit cleanly; `timeout 900s` terminated the lingering process with exit code `124`.
- A tokenized GitHub clone URL was written into `eval/eval_results.json` and `~/.repopilot/runs/4b44e8fe0149.json`; both local files were sanitized by replacing the token with `<redacted>`.
- Follow-up should prioritize either local/cached repo execution for eval or making `execute_fix` scrub credentials from error logs before persistence.

## 2026-06-12 Execute Error Token Redaction

Implemented the first follow-up from the live eval rerun: scrub tokenized GitHub clone URLs before execution errors are persisted.

Completed:

- Added `_redact_sensitive_error_text(...)` in `src/nodes/execute.py`.
- Redacts URLs shaped like `https://x-access-token:<token>@github.com/...` to `https://x-access-token:<redacted>@github.com/...`.
- Applied redaction to every `execute_fix(...)` path that writes `FixAttempt.error_log`:
  - patch apply output
  - test stdout/stderr output
  - execution exceptions, including `subprocess.TimeoutExpired` whose string contains the full command list

Tests added:

- `tests/test_new_agent.py::test_execute_fix_redacts_github_token_from_execution_error`
- `tests/test_new_agent.py::test_execute_fix_redacts_github_token_from_patch_apply_failure`
- `tests/test_new_agent.py::test_execute_fix_redacts_github_token_from_test_output`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_frame.py -q` -> 60 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/nodes/execute.py tests/test_new_agent.py`
- `ruff check src/nodes/execute.py tests/test_new_agent.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Split clone/network failures out of generic `execution_error`, likely as `failure_kind="infra_error"` or `failure_kind="clone_failed"`.
2. Route infra failures without consuming normal agent debug retries, so eval reports infrastructure blockage instead of root-cause/patch failure.

## 2026-06-12 Clone/Network Infra Error Classification

Implemented the next execution-layer follow-up: clone/network failures no longer consume normal debug retries or route through reflection.

Completed:

- `src/nodes/execute.py` now handles `git_clone(...)` failures separately from patch apply/test execution failures.
- Clone failures still use `test_result="execution_error"` for backward-compatible summary display, but now set `failure_kind="infra_error"`.
- Clone error logs still pass through `_redact_sensitive_error_text(...)`.
- Non-clone execution failures, such as `git apply` exceptions after a repo path exists, remain `failure_kind="execution_error"`.
- `src/nodes/verify.py` now routes `failure_kind="infra_error"` directly to `Phase.FAILURE` without incrementing `retry_count` or entering `REFLECT`.
- Infra failure reason is explicit: `Infrastructure error during execution: ...`.

Tests added/updated:

- `tests/test_new_agent.py::test_execute_fix_marks_clone_network_failure_as_infra_error`
- `tests/test_new_agent.py::test_verify_fix_infra_error_routes_to_failure_without_retry`
- Updated `tests/test_new_agent.py::test_execute_fix_redacts_github_token_from_execution_error` to assert clone timeout is `infra_error` while still redacted.

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_frame.py tests/test_run_store.py tests/test_eval_report.py -q` -> 71 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/nodes/execute.py src/nodes/verify.py tests/test_new_agent.py`
- `ruff check src/nodes/execute.py src/nodes/verify.py tests/test_new_agent.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Add a cached/local repo execution path so eval can avoid fresh GitHub clone on every attempt.
2. Then rerun the tox sample to reach patch apply/test behavior instead of infra clone failure.

## 2026-06-12 Cached Local Repo Execution Path

Implemented the minimum repository clone cache for `execute_fix(...)`.

Completed:

- `src/nodes/execute.py` now caches repositories under:
  - `${REPOPILOT_HOME}/repos/{owner}-{repo}` when `REPOPILOT_HOME` is set
  - otherwise `~/.repopilot/repos/{owner}-{repo}`
- `git_clone(state)` behavior:
  - cache hit: clone a temporary working copy from the local cache with `git clone --local --no-hardlinks`
  - cache miss: remote clone into the cache path, scrub the cached repo's `origin` URL to the non-token GitHub URL, then local-clone a temporary working copy
- Patches and tests run only in the temporary working copy, not in the cache repository.
- Cache population failures keep the existing infra-error path through `execute_fix(...)`, with token redaction preserved.
- Failed cache population removes the partial cache directory before raising.

Tests added:

- `tests/test_new_agent.py::test_git_clone_uses_cached_repo_without_remote_clone`
- `tests/test_new_agent.py::test_git_clone_populates_cache_then_clones_worktree`
- `tests/test_new_agent.py::test_git_clone_failed_cache_population_redacts_token`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_frame.py tests/test_run_store.py tests/test_eval_report.py -q` -> 74 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/nodes/execute.py tests/test_new_agent.py`
- `ruff check src/nodes/execute.py tests/test_new_agent.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Rerun the tox sample once to populate/use the repo cache:
   - `timeout 900s python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
2. If the run still hits infra clone failure on cache miss, retry once after cache population or manually prewarm `~/.repopilot/repos/tox-dev-tox`.
3. After clone is stable, inspect replay for patch apply/test behavior and the envpython hypothesis drift fix.

## 2026-06-12 Live Tox Eval Rerun With Repo Cache

Reran the tox sample after adding cached local repo execution:

- Command: `timeout 900s python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Sample: `tox-dev/tox#3075:3748`
- Run/trace id: `4248456c229f`
- Final phase: `FAILED`
- Error: `Maximum retries reached: 1.`
- Token used: `21116`
- Turns: `14`
- Report regenerated with `python3 eval/report.py`.
- Cache path confirmed: `~/.repopilot/repos/tox-dev-tox`

Diagnosis:

- Repo cache worked: `execute_fix` no longer spent minutes failing GitHub clone.
  - First `execute_fix` took 11.6s.
  - Later `execute_fix` calls failed instantly at patch apply.
- The run reached the intended patch/reflection loop:
  - 3 fix attempts
  - all 3 had `failure_kind="patch_apply_failed"`
  - errors were malformed/corrupt/non-applying unified diffs, not infra clone failures
- The envpython/root-cause hypothesis drift fix held in this run:
  - `df_0001` selected `H1`: cross-reference expansion applies `{envpython}` too early in the source environment context
  - `df_0002` reflection preserved `H1`
  - `df_0003` plan preserved `H1`
  - `df_0004` reflection preserved `H1`
  - `df_0005` plan preserved `H1`
  - `decision_warnings` was empty because the model did not drift and no defensive correction was needed
- The current top blocker is now patch generation quality, not clone infra and not hypothesis drift.

Operational notes:

- `eval/eval_results.json`, `eval/eval_summary.md`, and `examples/traces/case_1.json` were updated.
- Token search over the new eval artifacts and saved run found no tokenized GitHub URL leaks.
- The eval process still printed saved results but did not exit cleanly; `timeout 900s` killed the lingering process with exit code `124`.

Next recommended step:

1. Add patch apply preflight/repair before consuming retries:
   - run `git apply --check` on generated patch
   - if it fails, either ask planner for strict unified diff repair or add a deterministic patch-format validator before execution
2. Keep the existing hypothesis context guard; it worked for this run.
3. Separately fix eval async/client cleanup so runs exit without `timeout`.

## User Preference

The user wants local context saved so Codex does not spend a long time re-discovering prior state. Keep this file short, update it after meaningful milestones, and prefer concrete paths, commands, and current decisions over broad summaries.

## 2026-06-12 Patch Preflight Repair And Eval Cleanup

Implemented the two follow-ups from the cached tox eval run using subagents.

Completed:

- Patch preflight/repair:
  - `src/nodes/execute.py` now labels `git apply --check` failures as `Patch preflight check failed: ...`.
  - Post-check `git apply` failures are labeled separately as `Patch apply failed after preflight passed: ...`.
  - Both still preserve backward-compatible `test_result="patch_apply_failed"` and `failure_kind="patch_apply_failed"`.
  - `src/nodes/plan.py` and `src/nodes/reflect.py` prompts now call out preflight failures, include the exact preflight/apply error, include previous patch context, require strict complete unified diff repair, and keep the selected hypothesis anchored.
  - `src/nodes/verify.py` lets preflight repair failures avoid semantic `retry_count`, but now bounds consecutive preflight repair loops at `max_retries + 1`; after that it fails with `Patch preflight repair budget exhausted after N failures.`.
- Eval cleanup:
  - `src/http_client.py` now exposes `close_llm_client()` to await-close and clear the shared async LLM client.
  - `eval/agent_v2_harness.py::run_agent_v2_eval(...)` closes shared LLM resources in a `finally` block.
  - Cleanup errors are printed as warnings and do not mask successful eval results.

Subagents used:

- Hypatia handled patch preflight/repair behavior and tests.
- Poincare handled eval async client cleanup and tests.
- Main session added the bounded preflight repair guard after review.

Tests added/updated:

- `tests/test_patch_preflight.py`
- `tests/test_patch_retry.py`
- `tests/test_patch_repair_prompt.py`
- `tests/test_agent_v2_eval.py`
- `tests/test_http_client.py`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py tests/test_agent_v2_eval.py tests/test_http_client.py -q` -> 39 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_frame.py tests/test_run_store.py tests/test_eval_report.py tests/test_agent_v2_eval.py tests/test_http_client.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py -q` -> 113 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/http_client.py eval/agent_v2_harness.py src/nodes/execute.py src/nodes/verify.py src/nodes/plan.py src/nodes/reflect.py tests/test_agent_v2_eval.py tests/test_http_client.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py`
- `ruff check src/http_client.py eval/agent_v2_harness.py src/nodes/execute.py src/nodes/verify.py src/nodes/plan.py src/nodes/reflect.py tests/test_agent_v2_eval.py tests/test_http_client.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Rerun the tox live eval without wrapping it in `timeout`:
   - `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
2. Confirm whether the process exits cleanly after writing results.
3. Inspect replay for whether preflight repair reaches an apply-able patch or now fails with bounded patch-repair exhaustion.

## 2026-06-12 Live Eval After Cleanup + Memory Drain Fix

Continued from the previous recommendation and ran the tox live eval without `timeout`:

- Command: `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Sample: `tox-dev/tox#3075:3748`
- Run/trace id: `33ea071652b1`
- Final phase: `FAILED`
- Error: `Phase plan_fix timed out after 150.0s`
- Token used: `5601`
- Turns: `14`
- Updated artifacts:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`

Result:

- The eval process exited naturally with code `0` after writing results. This validates that the previous shared HTTP client cleanup addressed the main non-exiting eval process problem.
- This run did **not** reach patch preflight/repair. The first `plan_fix` node hit the 150s phase timeout before producing a plan.
- Replay/report now show a `node_diagnostic` entry for `plan_fix` with `event="phase"`, `status="timeout"`, `error_type="TimeoutError"`.

New cleanup issue found and fixed:

- The process printed a memory warning on exit:
  - `background memory write failed`
  - `sqlite3.ProgrammingError: Cannot operate on a closed database.`
- Root cause: `handle_failure` schedules `RepoStore.record_issue(...)` via `_fire_and_forget(...)`; eval cleanup called `close_store()` before the background SQLite write finished.
- Fix in `src/memory/repo_store.py`:
  - `_fire_and_forget(...)` now tracks pending background tasks in `_pending_background_tasks`.
  - `close_store()` now drains pending background tasks before closing cached SQLite connections.
- Added regression test:
  - `tests/test_memory.py::test_close_store_waits_for_pending_background_writes`

Verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_memory.py::test_close_store_waits_for_pending_background_writes -q`
- `python3 -B -m pytest -p no:cacheprovider tests/test_memory.py tests/test_agent_v2_eval.py tests/test_http_client.py -q` -> 39 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py -q` -> 10 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_eval_report.py -q` -> 2 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py tests/test_decision_frame.py tests/test_run_store.py tests/test_eval_report.py tests/test_agent_v2_eval.py tests/test_http_client.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py tests/test_memory.py -q` -> 125 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/memory/repo_store.py src/memory/__init__.py src/http_client.py eval/agent_v2_harness.py src/nodes/execute.py src/nodes/verify.py src/nodes/plan.py src/nodes/reflect.py tests/test_memory.py tests/test_agent_v2_eval.py tests/test_http_client.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py`
- `ruff check src/memory/repo_store.py src/memory/__init__.py src/http_client.py eval/agent_v2_harness.py src/nodes/execute.py src/nodes/verify.py src/nodes/plan.py src/nodes/reflect.py tests/test_memory.py tests/test_agent_v2_eval.py tests/test_http_client.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Operational note:

- In the restricted Codex sandbox, `aiosqlite` write paths can hang even for a minimal `RepoStore.record_file(...)` call. The same command completes immediately outside the sandbox. For memory/SQLite tests in this environment, run pytest with escalation/non-sandbox.
- Several sandbox-hung SQLite verification processes from this session were killed after confirming the hang was environmental.

Next recommended step:

1. Treat `plan_fix` timeout as the next live-eval blocker. The current LLM request timeout/retry window can still exceed or collide with the 150s phase timeout.
2. Add either stricter per-node LLM timeout alignment/cancellation diagnostics or a planner prompt/output-size reduction slice before rerunning the tox sample.
3. After `plan_fix` returns reliably, rerun the tox sample to validate whether patch preflight repair reaches an apply-able patch or bounded preflight exhaustion.

## 2026-06-12 Plan Fix Timeout Reliability

Implemented the plan-fix timeout reliability slice from:

- `docs/superpowers/plans/2026-06-12-plan-fix-timeout-reliability.md`

Completed:

- Added explicit LLM timeout budget helpers in `src/http_client.py`:
  - `LLM_RETRY_BACKOFF_MAX_SECONDS = 20.0`
  - `llm_retry_budget_seconds()`
- Aligned graph phase timeout policy:
  - `PHASE_TIMEOUTS["plan_fix"] = 180.0`
  - `PHASE_TIMEOUTS["reflect_on_failure"] = 180.0`
  - `tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window` now checks the actual retry budget plus planner margin.
- Added planner prompt diagnostics:
  - `plan_fix` records `node_diagnostics` event `prompt_built` before the LLM call.
  - Diagnostic includes `prompt_tokens_estimate`, `relevant_file_count`, `issue_body_chars`, `previous_failure_count`, and reflection/hypothesis-context flags.
  - Phase timeout replay can now show prompt size even if the LLM call is cancelled by the outer node timeout.
- Reduced planner prompt pressure in `src/nodes/plan.py`:
  - issue body limit: 2500 chars
  - file content limit: 1200 chars
  - max files in prompt: 2
  - previous failure log limit: 1000 chars
- Updated `eval/report.py` to summarize planner phase timeout diagnostics as:
  - `Planner timeout: plan_fix exceeded ...s.`

Live eval result after this slice:

- Command: `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Sample: `tox-dev/tox#3075:3748`
- Run/trace id: `6a6d488bc308`
- Final phase: `FAILED`
- Error: `Patch preflight repair budget exhausted after 3 failures.`
- Token used: `19193`
- Turns: `14`
- Process exited naturally with code `0`.
- Updated artifacts:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`

Diagnosis:

- The previous `plan_fix` timeout blocker is resolved for this run:
  - first `plan_fix`: success in ~76.4s
  - second `plan_fix`: success in ~98.7s
  - third `plan_fix`: success in ~65.6s
- Replay includes `prompt_built` diagnostics with prompt estimates around 2600 tokens on later repair attempts.
- The run now reaches the intended bounded patch preflight repair path:
  - first failure: patch did not apply to `src/tox/config/loader/ini/replace.py`
  - second failure: corrupt patch at line 22
  - third failure: corrupt patch at line 9
  - final failure: preflight repair budget exhausted after 3 failures
- No credential leaks found in eval artifacts, trace, or context using the local credential-pattern scan.

Verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_http_client.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_eval_report.py -q` -> 97 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/http_client.py src/graph.py src/nodes/plan.py eval/report.py tests/test_http_client.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_eval_report.py`
- `ruff check src/http_client.py src/graph.py src/nodes/plan.py eval/report.py tests/test_http_client.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_eval_report.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Focus on deterministic patch generation/repair quality. The live blocker is now malformed or non-applying unified diffs, not planner latency.
2. Consider adding a patch repair node/tool that converts a model patch plus `git apply --check` error into a validated replacement diff before consuming another LLM planning cycle.
3. Keep hypothesis continuity guard; the latest run preserved the `{envpython}` root-cause hypothesis while repair attempts failed at patch format/context.

## 2026-06-12 Deterministic Patch Repair Slice

Implemented a minimum deterministic patch repair layer before normal patch-failure reflection.

Completed:

- Added `src/patch_repair.py`.
  - `repair_unified_diff(patch)` returns `PatchRepair(patch, changed, reasons)`.
  - Repairs are intentionally syntax-only:
    - extract the first `diff --git ...` block from fenced/prose LLM output
    - remove invalid `index ...` lines, e.g. non-hex placeholder SHAs
    - recount unified diff hunk lengths from actual hunk body lines
    - ensure repaired diffs end with a newline
- `src/nodes/execute.py` now uses `apply_patch_with_repair(...)`.
  - Original patch is still checked first with `git apply --check`.
  - If original preflight fails and deterministic repair changes the patch, RepoPilot runs `git apply --check` on the repaired patch.
  - If repaired preflight passes, RepoPilot applies the repaired patch and records the repaired patch on `FixAttempt.patch_content` and `AgentState.patch_content`.
  - If repaired preflight still fails, the error log includes both original preflight output and repaired preflight output.
  - Existing `apply_patch(...) -> tuple[bool, str]` remains as a compatibility wrapper.
- Repair attempts are now visible to replay/report consumers via:
  - `_record_tool(..., "patch_repair", ...)`
  - `node_diagnostics` event `execute_fix / patch_repair` with status `success` or `error`, repair reasons, patch sizes, and output preview.

Tests added/updated:

- `tests/test_patch_preflight.py::test_repair_unified_diff_extracts_diff_and_recounts_hunk_lengths`
- `tests/test_patch_preflight.py::test_execute_fix_uses_repaired_patch_when_preflight_repair_passes`
- `tests/test_patch_preflight.py::test_execute_fix_records_repair_attempt_when_repaired_preflight_fails`
- Updated `tests/test_new_agent.py` execute tests to patch `apply_patch_with_repair(...)` and use `PatchApplyResult`.

Verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_run_store.py tests/test_eval_report.py -q` -> 93 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/patch_repair.py src/nodes/execute.py tests/test_patch_preflight.py tests/test_new_agent.py`
- `ruff check src/patch_repair.py src/nodes/execute.py tests/test_patch_preflight.py tests/test_new_agent.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`
- Credential-pattern scan over eval artifacts, trace, and context found no matches.

Live eval after initial deterministic repair integration:

- Command: `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Run/trace id: `43c33fe8fc89`
- Final phase: `FAILED`
- Error: `Patch preflight repair budget exhausted after 3 failures.`
- Token used: `18227`
- Turns: `15`
- Process exited naturally with code `0`.
- Updated artifacts:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`

Diagnosis:

- Deterministic repair did trigger on the first attempt:
  - original error: `error: corrupt patch at line 8`
  - repair reason: `recounted_hunk_lengths`
  - repaired preflight then failed as a real context/apply problem: `src/tox/tox_env/api.py: patch does not apply`
- This is useful progress: one malformed-diff error was converted into a concrete non-applying-hunk error.
- Later attempts still failed:
  - attempt 2: patch did not apply to `src/tox/tox_env/api.py:175`
  - attempt 3: corrupt patch at line 13
- The final blocker remains patch generation/repair quality.
- Additional finding: this live run's latest selected hypothesis drifted to an environment-name/hyphen factorization claim rather than the earlier `{envpython}` hypothesis. Treat hypothesis continuity as still worth monitoring.
- Note: `execute_fix` patch-repair node diagnostics were added after this live run. Rerun live eval to see `execute_fix / patch_repair` diagnostics in replay artifacts.

Next recommended step:

1. Rerun the tox live eval once with the new `execute_fix / patch_repair` diagnostics if replay visibility is needed.
2. Improve deterministic repair for context failures, likely by adding a local hunk-context resolver:
   - parse target file path and removed lines from the repaired diff
   - locate the removed line block in the current working tree
   - rewrite hunk start/count with correct nearby context
   - run `git apply --check` again before handing control back to reflection
3. Separately tighten hypothesis continuity beyond patch-format failures, because latest live run drifted away from the prior `{envpython}` line of reasoning.

## 2026-06-12 Search/Replace Patch Edits

User redirected the patch strategy away from LLM-authored unified diffs:

> Claude Code / Codex / Hermes patch tools use search/replace blocks; the tool should do deterministic string matching/replacement instead of asking the LLM to control diff syntax.

Implemented the minimum search/replace edit path:

- Added `PatchEdit` to `src/state.py` with:
  - `file_path` (also accepts LLM aliases `file` or `path`)
  - `search`
  - `replace`
  - `replace_all=False`
- Added `AgentState.patch_edits` and `FixAttempt.patch_edits`.
- `PlanDecision` now accepts `patch_edits`; `patch` is now default `""` for compatibility.
- `plan_fix(...)` prompt now asks for `patch_edits` first and treats legacy `patch` unified diff as fallback only.
- `plan_fix(...)` routes to `EXECUTE` when either `patch_edits` or legacy `patch` is present.
- `reflect_on_failure(...)` now tells the next plan to repair failed patch attempts as exact search/replace `patch_edits`, not strict unified diff repair.
- Failed `patch_edits` are included in the reflection prompt as `Patch Edits Applied` so the next planner can repair the exact search block.
- `execute_fix(...)` now applies `patch_edits` via deterministic exact replacement before any unified-diff fallback:
  - validates all edits before writing
  - rejects absolute paths, `..`, missing files, zero matches, and ambiguous multiple matches unless `replace_all=true`
  - records `apply_patch_edits` tool calls and `execute_fix / patch_edits` diagnostics
  - preserves old `apply_patch_with_repair(...)` unified-diff path for legacy responses
- `verify_fix(...)` treats both `git apply --check` failures and `Search/replace edit failed` as patch-quality failures, not semantic test failures.
- Patch repair failures are bounded by `max_retries + 1` with final failure reason:
  - `Patch repair budget exhausted after N failures.`

Tests added/updated:

- `tests/test_decision_schemas.py::test_plan_decision_accepts_search_replace_patch_edits`
- `tests/test_decision_frame.py::test_plan_fix_records_search_replace_patch_edits`
- `tests/test_patch_preflight.py::test_execute_fix_applies_search_replace_patch_edits_without_git_apply`
- `tests/test_patch_preflight.py::test_execute_fix_search_replace_failure_does_not_modify_file`
- `tests/test_patch_retry.py::test_consecutive_search_replace_failures_do_not_consume_semantic_retry`
- `tests/test_patch_retry.py::test_search_replace_repair_failures_are_bounded_without_semantic_retry`
- `tests/test_patch_repair_prompt.py::test_reflect_search_replace_failure_prompt_includes_failed_edits`
- Updated patch-repair prompt tests to expect search/replace guidance instead of unified-diff guidance.

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_patch_repair_prompt.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_decision_schemas.py tests/test_run_store.py tests/test_eval_report.py -q` -> 109 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/state.py src/schemas.py src/new_agent.py src/nodes/plan.py src/nodes/reflect.py src/nodes/execute.py src/nodes/verify.py tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_patch_preflight.py tests/test_patch_repair_prompt.py tests/test_patch_retry.py`
- `ruff check src/state.py src/schemas.py src/new_agent.py src/nodes/plan.py src/nodes/reflect.py src/nodes/execute.py src/nodes/verify.py tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_patch_preflight.py tests/test_patch_repair_prompt.py tests/test_patch_retry.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Rerun the tox live eval to see whether the planner now emits `patch_edits` and reaches apply/test behavior without unified diff failures:
   - `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
2. Inspect replay for `apply_patch_edits` tool calls and `execute_fix / patch_edits` diagnostics.
3. If `patch_edits` fail due to missing/ambiguous search blocks, improve planner prompt examples or add a deterministic nearby-context/fuzzy match suggestion, but keep exact replacement as the default write path.
4. Continue monitoring hypothesis continuity because run `43c33fe8fc89` drifted away from the earlier `{envpython}` hypothesis.

## 2026-06-12 Live Eval After Search/Replace Patch Edits

Reran the tox sample after switching planner/executor contract to `patch_edits`:

- Command: `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Sample: `tox-dev/tox#3075:3748`
- Run/trace id: `eb3000dc6490`
- Final phase: `FAILED`
- Error: `Token budget exceeded before code location.`
- Turns: `86`
- Token used: `50109`
- Report regenerated with `python3 eval/report.py`.
- Updated artifacts:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`
  - saved run: `~/.repopilot/runs/eb3000dc6490.json`

Diagnosis:

- This run did not reach `execute_fix`.
- It did not validate `patch_edits` execution because no plan emitted `patch_edits`.
- Replay has 7 plan frames:
  - `df_0001` through `df_0007`
  - all recommended `collect_more_context`
  - all routed to `locate_code`
- `locate_code` repeatedly returned the same 13 candidates / top 6 files, so the agent loop spent the budget on repeated `PLAN -> LOCATE -> PLAN`.
- Latest frame `df_0007` still recommended `collect_more_context`.
- `decision_warnings`: 7 warnings, all expected `LOCATE` but actual `PLAN`.
- `node_diagnostics`: 14 entries, all `plan_fix` prompt/LLM successes.
- No `apply_patch_edits` tool call and no `execute_fix / patch_edits` diagnostic appeared.
- Credential scan over eval artifacts, trace, and context found no token leaks.

Important new blocker:

- The current top blocker is now bounded context collection / locator expansion, not patch syntax.
- `collect_more_context` currently loops back to `locate_code`, but `locate_code` does not use `DecisionFrame.next_checks` to expand or vary search terms.

Next recommended step:

1. Add a guard for repeated `collect_more_context` frames when `locate_code` returns effectively unchanged context.
2. Use `DecisionFrame.next_checks` from the latest plan frame to seed additional code search/read targets.
3. Add a per-run cap such as `context_collection_count` or consume-frame repeat detection, then either ask the planner to commit to `patch_edits` with current context or fail with a clearer reason.
4. After that, rerun the same tox sample and check whether it reaches `apply_patch_edits`.

## 2026-06-12 BM25 Reranking

Implemented BM25 as the first locator-quality follow-up before the broader `collect_more_context` loop fix.

Completed:

- Added `src/retrieval.py`.
  - Pure-Python BM25, no new dependency.
  - Tokenizes issue query, file path, and file content.
  - Stops common English function words.
  - Returns detailed `BM25Score` rows via `bm25_scores(...)`.
  - Returns reranked `FileInfo` copies via `bm25_rerank(...)`.
  - Blends normalized BM25 score into existing `FileInfo.relevance_score`.
  - Appends reason text like `bm25 rerank score=...; matched issue terms: ...`.
  - Preserves original ordering/scores when there is no lexical signal.
  - Preserves original behavior for empty-content files.
- Integrated BM25 into `src/nodes/locate.py` after candidate files are hydrated.
  - Existing flow remains: GitHub Code Search / memory candidates -> path heuristic -> read top files.
  - New flow: hydrated files -> BM25 rerank -> `state.relevant_files`.
  - Records a `bm25_rerank` tool call with:
    - `query`
    - `candidate_count`
    - `applied`
    - ranked paths
    - raw and normalized BM25 scores
    - final relevance scores
    - matched terms

Tests added:

- `tests/test_retrieval.py::test_bm25_rerank_prioritizes_file_with_issue_terms`
- `tests/test_retrieval.py::test_bm25_rerank_preserves_order_when_no_useful_matches`
- `tests/test_retrieval.py::test_bm25_rerank_preserves_order_when_content_is_empty`
- `tests/test_locate_bm25.py::test_locate_code_records_bm25_rerank_and_reorders_hydrated_files`
- `tests/test_locate_bm25.py::test_locate_code_preserves_hydrated_order_when_bm25_has_no_signal`

Fresh verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_retrieval.py tests/test_locate_bm25.py -q` -> 5 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_retrieval.py tests/test_locate_bm25.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_tools.py -q` -> 81 passed
  - Needed escalation in Codex sandbox because `tests/test_tools.py` writes/cleans `~/.repopilot/cache`.
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/retrieval.py src/nodes/locate.py tests/test_retrieval.py tests/test_locate_bm25.py`
- `ruff check src/retrieval.py src/nodes/locate.py tests/test_retrieval.py tests/test_locate_bm25.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Next recommended step:

1. Continue with the `collect_more_context` loop fix:
   - feed latest `DecisionFrame.next_checks` into locator search/read terms
   - detect unchanged locate signatures
   - cap repeated context collection
2. Then rerun the tox sample and check whether BM25 plus next-check search reaches `patch_edits`.

## 2026-06-13 Live Eval Iteration Notes

The live tox sample was rerun several times from the real repo working tree at `/mnt/e/hermes-work/repopilot`.
Important: these runs intentionally used uncommitted working-tree changes. Before reruns, confirm the expected grep hits rather than assuming a committed checkout:

- `grep -n plan_fix src/graph.py` should show `plan_fix: 300.0`.
- `grep -n PLAN_FILE_CONTENT_LIMIT src/nodes/plan.py` should show `PLAN_FILE_CONTENT_LIMIT = 6000`.
- `grep -n "carried" src/nodes/locate.py` should hit the carried-context logic.
- `grep -n "_is_doc_file" src/nodes/locate.py` should hit doc filtering.
- `grep -n "_context_pressure" src/nodes/plan.py` should hit `_context_pressure_instructions`.

Live eval command used:

- `python3 -B eval/harness.py --agent-v2 --samples 1`
- After each run: `python3 -B eval/report.py`
- Main artifacts:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`

Recent run sequence:

- `1d718ae979db`
  - Working tree had `PLAN_FILE_CONTENT_LIMIT = 6000`, but `plan_fix` still timed out at the old 180s limit.
  - Result: `FAILED`, 42 turns, 37350 tokens.
  - Error: `Phase plan_fix timed out after 180.0s`.
  - Path: collect context twice, execute once, verify, reflect, then timeout on replanning.

- `e18641d0e896`
  - Confirmed `plan_fix = 300.0`.
  - Result: `FAILED`, 54 turns, 40816 tokens.
  - Error: `Context collection made no progress after 3 attempts.`
  - Path: `PLAN -> LOCATE` repeated for three `collect_more_context` frames, then `df_0004` recommended `stop`, followed by `FAILURE -> FAILED`.

- `00a81a299b89`
  - Confirmed `grep "carried" src/nodes/locate.py` hit.
  - Result: `FAILED`, 53 turns, 47530 tokens.
  - Error: `Context collection made no progress after 3 attempts.`
  - Path remained the context-collection loop; no execution.

- `3a473aedb01c`
  - Confirmed `grep "_is_doc_file" src/nodes/locate.py` hit.
  - Result: `FAILED`, 49 turns, 45070 tokens.
  - Error: `Context collection made no progress after 3 attempts.`
  - Path remained the context-collection loop; no execution.
  - Locate candidate counts changed across rounds: `8 -> 13 -> 16 -> 17`, with top ranked `6 -> 6 -> 5 -> 2`, so doc filtering/carrying changed retrieval behavior but did not break the loop.

- `9e043fe7f8b7`
  - Confirmed `grep "_context_pressure" src/nodes/plan.py` hit.
  - Result: `FAILED`, 45 turns, 53834 tokens.
  - Error: `Token budget exceeded before execution.`
  - This was the first recent run that escaped the pure context loop:
    - `df_0001`, `df_0002`, `df_0003`: `plan -> collect_more_context`.
    - `df_0004`: `plan -> execute`, executor applied 5 patch edits.
    - `verify_fix` failed and routed to `reflect_on_failure`.
    - `df_0005`: `reflect -> plan`.
    - `df_0006`: `plan -> execute`, planner proposed 2 more edits.
    - Second `execute_fix` failed before applying because the run exceeded token budget.
  - Final route path:
    - `LOCATE -> PLAN -> LOCATE -> PLAN -> LOCATE -> PLAN -> LOCATE -> PLAN -> EXECUTE -> VERIFY -> REFLECT -> PLAN -> EXECUTE -> FAILURE -> FAILED -> __end__`.

Current interpretation:

- `_context_pressure_instructions` improved behavior enough to force a patch attempt instead of stopping after repeated context collection.
- The next blocker is token budget exhaustion after reflection/replanning, not retrieval alone.
- The latest artifacts currently correspond to run `9e043fe7f8b7`.

Next recommended step:

1. Inspect `examples/traces/case_1.json` for the exact `df_0004` and `df_0006` plans, plus `node_diagnostics` and failed verification reason.
2. Decide whether to reduce prompt/context size after reflection or increase eval token budget for this live sample.
3. If reducing context, preserve the useful carried files but trim repeated issue/context text and prior frame detail before the second planner call.
4. Rerun the tox sample and check whether the second `execute_fix` can apply edits and reach `verify_fix`.

## 2026-06-13 Scrapy Sample First-Green Attempt

Target sample:

- `scrapy/scrapy#6195:7095`
- Command: `python3 eval/harness.py --agent-v2 --sample-id 'scrapy/scrapy#6195:7095' --max-retries 2 --token-budget 100000`

Preflight checks requested by user:

- `grep -n _pip_install_editable src/nodes/execute.py` hit.
- `grep -n sample-id eval/harness.py` hit.

Small harness fix made before live eval:

- The first command failed immediately with `TypeError: run_agent_v2_eval() got an unexpected keyword argument 'sample_id'`.
- Root cause: `eval/agent_v2_harness.run_agent_v2_eval(...)` accepted `sample_id`, but the wrapper in `eval/harness.py` did not.
- Fixed `eval/harness.py::run_agent_v2_eval(...)` to accept and forward `sample_id`.
- Added `tests/test_agent_v2_eval.py::test_harness_run_agent_v2_eval_forwards_sample_id`.
- Verification passed:
  - `python3 -B -m pytest -p no:cacheprovider tests/test_agent_v2_eval.py -q` -> 12 passed
  - `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile eval/harness.py tests/test_agent_v2_eval.py`
  - `ruff check eval/harness.py tests/test_agent_v2_eval.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Live eval runs:

- `08e603a56c18`
  - Result: `FAILED`
  - Error: infra clone timeout while populating `/home/morganbest/.repopilot/repos/scrapy-scrapy`.
  - The timed-out cache dir was a partial git repo with no valid `HEAD` and only about 112K.
  - Moved it to `/home/morganbest/.repopilot/repos/scrapy-scrapy.broken-08e603a56c18`.
  - Manually prewarmed cache with `git clone --depth 1 --filter=blob:none --single-branch https://github.com/scrapy/scrapy.git /home/morganbest/.repopilot/repos/scrapy-scrapy`.
  - Verified cache `HEAD` as `af30cfe`.

- `f993791c1b7b`
  - Result: `FAILED`
  - Error: `Patch repair budget exhausted after 4 failures.`
  - Turns: 22
  - Token used: 55099
  - First plan produced 1 `patch_edit` and `execute_fix` applied it.
  - First verification failed with `ModuleNotFoundError: No module named 'w3lib'`.
  - Saved run showed `_pip_install_editable` did run but returned `success=false` for `python3 -m pip install -e .`.
  - Manual dry-run in the retained worktree reproduced the install failure as PEP 668 `externally-managed-environment`.
  - Later reflection/planning cycles preserved the correct BOM-stripping root cause but repeatedly failed exact search/replace matching in `scrapy/downloadermiddlewares/robotstxt.py`.

Current interpretation:

- The sample is still a good first-green target, but the execution environment is the blocker.
- The current editable-install implementation uses system Python/pip, which fails on this machine because Python 3.14 is externally managed.
- Search/replace edit quality also needs improvement, but the first actual blocker was dependency installation: the initial patch applied and then pytest could not import Scrapy dependencies.

Next recommended step:

1. Change `execute_fix` to create a per-worktree virtualenv, e.g. `.repopilot-venv`.
2. Run editable install with the venv pip instead of system pip.
3. Run pytest with the venv on `PATH` / `VIRTUAL_ENV`, so commands like `pytest -k robotstxt` use installed deps.
4. Record venv install stdout/stderr preview in `pip_install_editable` diagnostics; current record only stores return code.
5. Rerun the same scrapy sample after this venv execution fix.

## 2026-06-13 Scrapy Sample Venv Follow-up

User added initial venv execution support and asked to rerun:

- Pre-run check: `grep -n _create_venv src/nodes/execute.py` hit.
- Scrapy cache remained prewarmed: `/home/morganbest/.repopilot/repos/scrapy-scrapy`, `HEAD=af30cfe`.

Run `7b7ed2e7abac`:

- Command: `python3 eval/harness.py --agent-v2 --sample-id 'scrapy/scrapy#6195:7095' --max-retries 2 --token-budget 100000`
- Result: `FAILED`
- Error: `Patch repair budget exhausted after 4 failures.`
- Diagnostics:
  - `create_venv`: `created=false`, reason `venv creation failed`
  - `pip_install_editable`: fell back to `python3`, `success=false`
  - `run_pytest`: used `/tmp/...-venv/bin/python` and failed with `No module named pytest`
- Root cause:
  - Local Python 3.14 lacks `ensurepip` / `python3.14-venv`.
  - `python3 -m venv --system-site-packages ...` leaves a partial venv with `bin/python` but no pip.
  - `run_pytest` used that partial venv because it only checked for `bin/python`.

Execution-layer fixes made:

- `_create_venv(...)` now writes a `.repopilot-ready` marker only after successful venv creation.
- `run_pytest(...)` uses the venv only when the ready marker exists, so partial venvs are ignored.
- `_create_venv(...)` now removes partial venv dirs and falls back to `uv venv --system-site-packages ...` when stdlib venv fails.
- Added `_ensure_pytest_available(...)`:
  - checks `import pytest, pytest_twisted` in the selected interpreter
  - installs `pytest pytest-twisted` into the venv when missing
- `run_pytest(...)` now rewrites explicit commands beginning with bare `pytest` to `<venv-python> -m pytest ...` when a ready venv exists, so it does not fall through to system `/home/morganbest/.local/bin/pytest`.

Tests added/updated:

- `tests/test_execute_install.py::test_create_venv_falls_back_to_uv_when_stdlib_venv_lacks_ensurepip`
- `tests/test_execute_install.py::test_run_pytest_ignores_partial_venv_without_ready_marker`
- `tests/test_execute_install.py::test_ensure_pytest_available_installs_runner_deps_when_missing`
- `tests/test_execute_install.py::test_ensure_pytest_available_skips_install_when_runner_deps_exist`
- `tests/test_execute_install.py::test_run_pytest_rewrites_bare_pytest_command_to_venv_python`

Verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_execute_install.py -q` -> 14 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_execute_install.py tests/test_patch_preflight.py tests/test_new_agent.py -q` -> 53 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/nodes/execute.py tests/test_execute_install.py`
- `ruff check src/nodes/execute.py tests/test_execute_install.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Run `8d54ceeb5800` after uv fallback:

- Result: `FAILED`
- Error: malformed/truncated planner JSON after several repair attempts.
- Useful diagnostics:
  - `create_venv`: `created=true`, `creator=uv`
  - `pip_install_editable`: `success=true`, used `<clone>-venv/bin/python`
  - pytest still failed with `ModuleNotFoundError: w3lib` because explicit command `pytest ...` resolved to system pytest, not the venv interpreter.

Run `b980ba240025` after pytest-runner fix:

- Result: final phase `FAILED`, but patch/test path succeeded.
- Final error: `Failed to push or create PR: Client error '404 Not Found' for url 'https://api.github.com/repos/scrapy/scrapy/git/refs'`.
- Key diagnostics:
  - `create_venv`: `created=true`, `creator=uv`
  - `pip_install_editable`: `success=true`, command `<clone>-venv/bin/python -m pip install -e .[test]`
  - `ensure_pytest_available`: `success=true`, command `<clone>-venv/bin/python -m pip install pytest pytest-twisted`
  - `apply_patch_edits`: `applied=true`
  - `run_pytest`: command `<clone>-venv/bin/python -m pytest tests/test_downloadermiddleware_robotstxt.py`, `returncode=0`
  - test output: `13 passed, 13 skipped in 0.11s`
- Actual edit applied:
  - file: `scrapy/downloadermiddlewares/robotstxt.py`
  - replaced `rp = self._parserimpl.from_crawler(self.crawler, response.body)`
  - with BOM-stripping body logic before passing body to the parser.

Current interpretation:

- The first apply/test green has been reached for the scrapy sample.
- `success=false` only because the graph continues into `commit_fix` and tries to push/create a PR against upstream `scrapy/scrapy`, which fails with GitHub API 404/permission behavior.

Next recommended step:

1. Add an eval-safe completion mode or commit policy:
   - when verification passes in eval/live benchmark mode, stop at `DONE` with patch/test evidence instead of attempting upstream PR creation; or
   - make `commit_fix` optional/configurable and treat PR creation failure separately from fix/test success.
2. Rerun `scrapy/scrapy#6195:7095` and confirm final payload reports success based on verified patch/test result.

## 2026-06-13 Scrapy Sample First Green

Implemented the eval-safe completion switch requested by the user:

- In `eval/agent_v2_harness.py::evaluate_agent_v2_sample(...)`, the `agent_v2(...)` call now passes `skip_commit=True`.
- Updated `tests/test_agent_v2_eval.py::test_evaluate_agent_v2_sample_saves_run_and_attaches_replay` to assert `skip_commit=True` is forwarded.

While rerunning the sample, a repeated planner-output compatibility issue appeared:

- Runs `e8e4fc5bbd21` and `3d3296610962` failed in `plan_fix` because the LLM emitted `decision_frame.hypotheses[0].score` as `9` / `9.0`.
- `Hypothesis.score` expected `0..1`, so `PlanDecision` validation crashed before execution.
- Added a narrow normalizer in `src/state.py::Hypothesis._normalize_score(...)`:
  - numeric scores in `(1, 10]` are treated as 10-point scores and divided by 10
  - other invalid values still fail normal Pydantic bounds
- Added `tests/test_decision_schemas.py::test_plan_decision_normalizes_hypothesis_score_from_ten_point_scale`.

Verification passed:

- `python3 -B -m pytest -p no:cacheprovider tests/test_agent_v2_eval.py -q` -> 12 passed
- `python3 -B -m pytest -p no:cacheprovider tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_agent_v2_eval.py -q` -> 63 passed
- `PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/state.py eval/agent_v2_harness.py tests/test_decision_schemas.py tests/test_agent_v2_eval.py`
- `ruff check src/state.py eval/agent_v2_harness.py tests/test_decision_schemas.py tests/test_agent_v2_eval.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`

Successful live eval:

- Command: `python3 eval/harness.py --agent-v2 --sample-id 'scrapy/scrapy#6195:7095' --max-retries 2 --token-budget 100000`
- Run/trace id: `68285b8d86a2`
- Final phase: `DONE`
- Payload success: `True`
- Error: `None`
- Turns: 19
- Token used: 11834
- Report now shows `agent_v2: 1/1 success`.
- Saved run state:
  - `skip_commit=True`
  - `create_venv`: `created=true`, `creator=uv`
  - `pip_install_editable`: `success=true`
  - `ensure_pytest_available`: `success=true`
  - `apply_patch_edits`: `applied=true`
  - `run_pytest`: `<clone>-venv/bin/python -m pytest tests/test_robotstxt_interface.py`, `returncode=0`
  - test output: `15 passed, 10 skipped in 0.08s`

Current interpretation:

- RepoPilot now has its first verified green live agent-v2 eval sample.
- The remaining reporting oddity is that legacy aggregate rows still show `patch_apply: 0.000` / `test_pass: N/A` for agent-v2 mode, while the agent-v2 summary correctly shows `1/1 success`.

## 2026-06-29 Legacy Test Cleanup + Eval Batch (4 samples)

Cleaned up a stale dead test and ran a small fresh eval batch.

Cleanup:

- Removed root-level `test_intelligent_agent.py`. It imported the removed v1
  API (`AgentContext`, `Reflection`) and iterated `AgentState` as an Enum,
  but `AgentState` is now a Pydantic `BaseModel`. It was also interactive
  (`input()`) and hit fake URLs / localhost. Not referenced by CI
  (`.github/workflows/ci.yml` runs `pytest tests/` only). The maintained
  `tests/` suite supersedes it.
- Full repo-root `pytest` is now clean (was 3 failed / 271 passed).

Eval batch (each `--agent-v2 --max-retries 2 --token-budget 100000`):

| Sample | Final phase | Failure layer |
|--------|-------------|---------------|
| `scrapy/scrapy#6195:7095` | DONE (green) | — (control reproduced) |
| `tox-dev/tox#3075:3748` | FAILED | fix-quality (same patch twice) |
| `aio-libs/aiomysql#792:839` | FAILED | convergence (no progress) |
| `celery/celery#9613:9614` | FAILED | infra (github clone network timeout) |

Bug found + fixed (schema-compat, same class as the earlier `score` fix):

- tox initially crashed in `plan_fix` with two `PlanDecision` validation
  errors: the LLM emitted `decision_frame.hypotheses[0].id = 1` (int) and
  `selected_hypothesis_id = 1` (int), but both fields require `str`.
- Fix in `src/state.py`:
  - `Hypothesis._coerce_id_to_str` (field_validator `id`, mode="before")
    coerces int/float ids to `str`.
  - `DecisionFrame._coerce_selected_hypothesis_id_to_str` (field_validator
    `selected_hypothesis_id`, mode="before") coerces int/float to `str`,
    leaves `None` alone.
- Test: `tests/test_decision_schemas.py::test_plan_decision_coerces_integer_hypothesis_ids_to_strings`.

Verification:

- `python3 -B -m pytest -p no:cacheprovider -q` -> 272 passed.
- `ruff check src/state.py tests/test_decision_schemas.py --select=E,F,I --ignore=E501` -> clean.
- Live tox re-run after fix: no more schema crash. It now progresses
  LOCATE -> PLAN -> EXECUTE -> VERIFY -> REFLECT -> PLAN and terminates with
  "Same patch produced the same failure twice." The failure point moved from
  an infrastructure/schema crash to a genuine fix-quality / convergence
  problem — forward progress.

Current interpretation:

- 1/4 green. Of the 3 failures: 1 fix-quality, 1 convergence, 1 network/infra.
- The remaining agent-capability frontier is unchanged: planner convergence
  + patch quality (the planner keeps re-proposing the same non-fixing patch,
  or loops on collect_more_context without new context).

Next recommended step:

- Decide between (a) attacking convergence/fix-quality directly (e.g. a
  last-round structural forcing of `recommended_action=execute`, or
  diversifying the patch on REFLECT instead of re-proposing the same edit),
  or (b) widening the eval batch to get a larger resolved% distribution
  first. Network-flaky samples (celery) should be retried or skipped, not
  counted as agent failures.
