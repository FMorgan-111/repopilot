# RepoPilot Architecture

_Last updated: 2026-06-15_

RepoPilot is a GitHub issue -> fix PR agent with an inspectable control loop. The current runtime is built around typed Pydantic state, explicit decision frames, and a white-box `plan -> reflect -> execute` loop so runs can be replayed, resumed, and audited.

## System at a Glance

```text
GitHub issue / CLI / API
    -> AgentState + Tracer
    -> LangGraph (or fallback runner)
    -> UNDERSTAND -> LOCATE -> PLAN -> EXECUTE -> VERIFY
    -> REFLECT -> PLAN loop or COMMIT / FAILURE
    -> JSON payload + trace log + optional paused run on disk
```

## Repository Map

| File / module | Role |
|---|---|
| `src/main.py` | FastAPI surface for `/analyze`, `/agent`, `/agent/v2`, `/agent/v2/resume`, and replay endpoints |
| `src/cli.py` | CLI entry points for issue runs, resume, run listing, inspection, and replay |
| `src/new_agent.py` | Orchestration layer that builds the graph, runs it, and shapes API payloads |
| `src/graph.py` | Routing logic, fallback graph runner, timeouts, and decision-frame consumption |
| `src/state.py` | Core state models, enums, helper functions, and traceable decision frame records |
| `src/schemas.py` | Structured LLM output schemas for planning and reflection |
| `src/nodes/` | Phase implementations: understand, locate, plan, execute, verify, reflect, commit, failure |
| `src/run_store.py` | Persistent storage for paused runs plus inspection/replay helpers |
| `src/tracer.py` | Structured trace event capture |

## Core Runtime Model

### `AgentState`

`AgentState` is the single source of truth for a run. It carries:

- issue metadata and repo metadata
- the current `Phase`
- ranked files, fix attempts, tool calls, and conversation history
- token usage, retry counters, and failure state
- white-box decision data such as `decision_frame`, `frame_history`, `route_decisions`, and `decision_warnings`
- pause state via `pending_human_input` and `human_input_request`

### Decision frames

[`src/state.py`](src/state.py) defines the shared reasoning snapshot used by planning and reflection:

- `DecisionFrame`
- `Hypothesis`
- `PatchEdit`
- `FixAttempt`
- `ToolCall`
- `FinalReport`

`DecisionFrame` is the inspectable protocol between nodes. It records:

- `stage` (`diagnose`, `plan`, `reflect`)
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

## Structured LLM Outputs

[`src/schemas.py`](src/schemas.py) validates full LLM responses, not just the embedded frame.

- `PlanDecision` wraps the complete PLAN output
- `ReflectDecision` wraps the complete REFLECT output
- both normalize older flat JSON formats for backward compatibility
- both require the embedded `decision_frame.stage` to match the node that produced it

This keeps the LLM contract explicit and makes legacy responses survivable while the prompt format evolves.

## Graph and Routing

RepoPilot uses LangGraph when it is installed, and a small pure-Python fallback graph when it is not. Both paths share the same state contract.

`build_agent_graph()` in [`src/new_agent.py`](src/new_agent.py) wires the nodes, while [`src/graph.py`](src/graph.py) owns the router.

### Execution flow

1. `agent_v2()` creates a fresh `AgentState` and trace id.
2. The graph runs `UNDERSTAND -> LOCATE -> PLAN -> EXECUTE -> VERIFY`.
3. A failed verify routes to `REFLECT`, then back to `PLAN`.
4. A successful verify routes to `COMMIT`, unless `skip_commit` is enabled for eval mode.
5. Hard failures route to `FAILURE` and then terminate.

### Routing rules

The router first tries to consume the newest `DecisionFrame`. If that frame is fresh and supported, its `recommended_action` decides the next phase.

Supported action -> phase mapping:

- `collect_more_context` -> `LOCATE`
- `plan` -> `PLAN`
- `execute` -> `EXECUTE`
- `reflect` -> `REFLECT`
- `stop` -> `FAILURE`
- `ask_user` -> `WAITING_FOR_USER`

If the frame is stale, missing, already consumed, or unsupported, routing falls back to `current_phase`. Those cases are recorded in `route_decisions` and `decision_warnings` instead of silently steering control flow.

### Human-in-the-loop pause

`ask_user` turns a run into a durable pause:

- `pending_human_input = True`
- `current_phase = WAITING_FOR_USER`
- `human_input_request` is populated from the frame, using `next_checks[0]` when present and falling back to the frame summary
- the run is saved so it can be resumed later

`resume_agent_v2(run_id, human_answer)` reloads the saved run, appends the answer to conversation history, clears the pause state, resets the phase to `PLAN`, and continues from there.

## Phase Responsibilities

| Phase node | Responsibility |
|---|---|
| `understand_issue` | Read the GitHub issue, classify the task, and extract useful signals |
| `locate_code` | Search and rank repository files that are likely relevant |
| `plan_fix` | Produce a patch-oriented plan and structured decision frame |
| `execute_fix` | Apply the patch and run the target test command |
| `verify_fix` | Interpret test output and decide success, reflection, or failure |
| `reflect_on_failure` | Analyze why the previous attempt failed and what to change next |
| `commit_fix` | Publish the fix and open a draft PR |
| `handle_failure` | Report partial progress and exit cleanly |

## Persistence and Replay

[`src/run_store.py`](src/run_store.py) persists paused runs under `~/.repopilot/runs` by default, or under `REPOPILOT_HOME` when that environment variable is set.

Saved runs support:

- `list_runs`
- `inspect_run`
- `replay_run`
- human-readable markdown replay output

The API payload returned by `agent_v2()` includes the full white-box surface:

- `decision_frame`
- `frame_history`
- `decision_warnings`
- `route_decisions`
- `node_diagnostics`
- `human_input_request`
- `waiting_for_user`
- `run_id`

This makes the runtime easy to inspect from both the CLI and HTTP API.

## Observability

[`src/tracer.py`](src/tracer.py) captures structured trace events. In addition, the saved state and replay output preserve routing and diagnostic metadata so failures can be inspected after the fact.

Important observability artifacts:

- `Tracer` events for run-level tracing
- `route_decisions` for route reconstruction
- `decision_warnings` for frame/router mismatches
- `node_diagnostics` for timeouts and crashes
- saved paused-run JSON for later resume or replay

## Compatibility Surface

- `/agent/v2` is the primary runtime endpoint
- `/agent/v2/resume` resumes paused runs
- `/agent/v2/runs/{run_id}/replay` exposes replay data as JSON or markdown
- `/analyze` and `/agent` remain for backward compatibility
- `intelligent_analyze_issue()` is an alias for `agent_v2()`

## Operational Boundaries

RepoPilot is intentionally narrow:

- it targets professional developers working on real repositories
- it prefers inspectable reasoning over black-box automation
- it uses bounded retries and token budgets instead of infinite loops
- it treats unsupported control actions as audit-only fallbacks
- it supports eval mode via `skip_commit`, where a verified fix can terminate without opening a PR

## Testing Coverage

The current test suite focuses on the white-box runtime:

- schema validation: `tests/test_decision_schemas.py`
- decision-frame routing and pause/resume: `tests/test_decision_frame.py`
- graph behavior and agent flow: `tests/test_new_agent.py`
- paused-run storage: `tests/test_run_store.py`
- HTTP endpoints: `tests/test_main.py`

## Related Docs

- [`docs/CODEX_CONTEXT.md`](docs/CODEX_CONTEXT.md)
- [`docs/PRODUCTION_PLAN.md`](docs/PRODUCTION_PLAN.md)
- [`docs/MEMORY_DESIGN_V2.md`](docs/MEMORY_DESIGN_V2.md)
- [`docs/RESUME_STRATEGY.md`](docs/RESUME_STRATEGY.md)
- [`docs/TECH_DESIGN_AND_INTERVIEW.md`](docs/TECH_DESIGN_AND_INTERVIEW.md)
