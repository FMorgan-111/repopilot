# Success-First Model Escalation and Constrained Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise RepoPilot's deterministic SWE-bench Verified internal success rate from 1/10 to at least 4/10 by retaining Gemini Flash as the default, escalating stalled samples to Claude Opus, and allowing the active model to request safe local evidence tools.

**Architecture:** Extend the existing exact-commit phase graph with deterministic Gemini→Opus routing, bounded evidence tools, a two-stage verified-edit flow, PatchGate, rolling attempt summaries, and a post-VERIFY COVERAGE phase. COVERAGE dynamically proves the same targeted test passes on the fix and fails on the base commit, generating a test through the active model when necessary. Persist every safe routing, summary, tool, and coverage decision in `AgentState` so saved runs and eval reports are replayable.

**Tech Stack:** Python 3.10+, Pydantic v2, httpx, pytest/pytest-asyncio, LangGraph fallback graph, Ruff, SWE-bench Verified harness.

## Global Constraints

- Never write API keys into source, tests, documentation, traces, run files, eval artifacts, shell history snippets, or Git configuration. Tests use sentinel strings only.
- Gemini remains `gemini-3.5-flash:stable`; Opus is `claude-opus-4-8:stable` and is enabled only when `REPOPILOT_ESCALATION_ENABLED=1` and an escalation key is present.
- Model code never chooses escalation. `ModelPolicy` is deterministic and escalation is one-way for a sample.
- Agent-selected tools are local, read-only except for the existing patch execution path, confined to `state.repo_path`, and cannot clone, checkout, access the network, push, create PRs, score benchmarks, or run arbitrary shell.
- No SWE-bench evaluator-only field (`patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`) may enter an agent prompt, state evidence record, trace, or prediction.
- Every completed PLAN→EXECUTE→VERIFY→REFLECT failure loop replaces one rolling, secret-redacted outcome summary of at most 200 Unicode characters; PLAN injects it exactly once.
- VERIFY success is not terminal. A task is internally successful only after COVERAGE records two fixed passes and two stable base assertion failures for the same targeted test.
- Test generation is test-only and limited to two attempts. A second invalid differential test terminates as `test_generation_failed` without a PR.
- Use RED/GREEN/REFACTOR for each task. Run the named focused test before implementation and observe the expected failure.
- Preserve all current tests and public call defaults unless a task explicitly changes them.
- Commit after every task only when that task's focused tests and Ruff checks pass.

---

### Task 1: Centralize RepoPilot home and repair memory placement

**Files:**

- Create: `src/home.py`
- Modify: `src/memory/repo_store.py`
- Modify: `eval/agent_v2_harness.py`
- Test: `tests/test_memory.py`
- Test: `tests/test_agent_v2_eval.py`

**Interfaces:**

```python
# src/home.py
def repopilot_home() -> Path:
    """Return expanded REPOPILOT_HOME or ~/.repopilot."""
```

`RepoStore(base_path=None)` must use `repopilot_home() / "memory"`. Existing explicit `base_path` behavior stays unchanged. The eval results fallback must call the same helper rather than duplicate environment parsing.

- [ ] **Step 1: Bootstrap the ignored development environment**

Run: `python -m venv .venv`

Run: `.venv/bin/python -m pip install -e '.[dev,eval]' build`

Expected: `.venv/bin/python`, pytest, Ruff, build, and optional SWE-bench adapters are available; `.venv/` remains ignored by Git.

- [ ] **Step 2: Add failing path-contract tests**

Add `test_repo_store_default_honors_repopilot_home` with a temporary `REPOPILOT_HOME`, instantiate `RepoStore()` and assert its database is under `<temp>/memory`. Add an eval fallback test asserting the helper-controlled `<temp>/eval/eval_results.json` path.

- [ ] **Step 3: Run the RED tests**

Run: `.venv/bin/python -m pytest tests/test_memory.py::test_repo_store_default_honors_repopilot_home tests/test_agent_v2_eval.py::test_run_agent_v2_eval_falls_back_when_results_path_write_fails -q`

Expected: the new memory assertion fails because `RepoStore` currently expands `~/.repopilot/memory` directly.

- [ ] **Step 4: Implement the shared home helper and switch both callers**

Use `Path(os.environ.get("REPOPILOT_HOME", "~/.repopilot")).expanduser()`. Do not create directories in `repopilot_home()` itself; callers retain ownership of directory creation.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_memory.py tests/test_agent_v2_eval.py -q`

Run: `.venv/bin/python -m ruff check src/home.py src/memory/repo_store.py eval/agent_v2_harness.py tests/test_memory.py tests/test_agent_v2_eval.py`

Commit: `fix: honor repopilot home for memory storage`

---

### Task 2: Add independent primary and escalation provider configuration

**Files:**

- Create: `src/model_provider.py`
- Modify: `src/http_client.py`
- Modify: `src/llm.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_model_provider.py`
- Test: `tests/test_http_client.py`
- Test: `tests/test_llm_json_tolerance.py`

**Interfaces:**

```python
ProviderName = Literal["primary", "escalation"]

class ModelConfig(BaseModel):
    provider: ProviderName
    model: str
    base_url: str
    api_key: SecretStr

def get_model_config(provider: ProviderName) -> ModelConfig: ...
def escalation_is_configured() -> bool: ...
def sanitize_provider_error(exc: BaseException) -> str: ...
```

```python
async def llm_request(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    *,
    provider: ProviderName = "primary",
    **kwargs: object,
) -> dict: ...

async def llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    *,
    provider: ProviderName = "primary",
    temperature: float = 0.2,
) -> dict: ...
```

Primary configuration keeps existing fallbacks: `LINOAPI_API_KEY`/`LLM_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL`. Escalation configuration uses only `LLM_ESCALATION_API_KEY`, `LLM_ESCALATION_BASE_URL`, and `LLM_ESCALATION_MODEL`; it must never fall back to the primary key implicitly.

- [ ] **Step 1: Write provider selection and redaction tests**

Cover separate URLs, models, and authorization headers with sentinel keys; missing escalation key; primary backward compatibility; independent shared clients keyed by base URL/provider; and sanitization of a provider exception containing bearer/query/header-like secrets.

- [ ] **Step 2: Add forwarding tests for `llm_call`**

Assert both the first JSON request and its JSON-repair retry preserve `provider="escalation"`. Update existing fakes to accept the new keyword-only provider argument.

- [ ] **Step 3: Run the RED tests**

Run: `.venv/bin/python -m pytest tests/test_model_provider.py tests/test_http_client.py tests/test_llm_json_tolerance.py -q`

Expected: imports/signatures fail because provider-aware configuration does not exist.

- [ ] **Step 4: Implement provider-aware requests**

Resolve config at request time, construct the authorization header locally, and never include it in raised messages. Keep retry, streaming, tool-call assembly, and empty-response behavior unchanged. Change the client cache/reset helper to close every provider client.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_model_provider.py tests/test_http_client.py tests/test_llm_json_tolerance.py tests/test_llm.py -q`

Run: `.venv/bin/python -m ruff check src/model_provider.py src/http_client.py src/llm.py tests/test_model_provider.py tests/test_http_client.py tests/test_llm_json_tolerance.py`

Commit: `feat: add isolated escalation model provider`

---

### Task 3: Persist model routing state and deterministic escalation policy

**Files:**

- Create: `src/model_policy.py`
- Modify: `src/state.py`
- Modify: `src/new_agent.py`
- Modify: `src/run_store.py`
- Test: `tests/test_model_policy.py`
- Test: `tests/test_run_store.py`
- Test: `tests/test_new_agent.py`

**State models (declared in `src/state.py` to keep saved-state types out of the provider layer):**

```python
class ModelInvocation(BaseModel):
    model: str
    provider: Literal["primary", "escalation"]
    node: str
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    status: Literal["ok", "invalid_response", "error"]
    error_class: str = ""

class NoProgressEvent(BaseModel):
    kind: str
    fingerprint: str
    node: str

# AgentState additions
active_model: str = "gemini-3.5-flash:stable"
active_provider: Literal["primary", "escalation"] = "primary"
escalated: bool = False
escalation_reason: str = ""
no_progress_rounds: int = 0
last_plan_signature: str = ""
last_context_fingerprint: str = ""
last_test_failure_signature: str = ""
model_history: list[ModelInvocation] = Field(default_factory=list)
no_progress_history: list[NoProgressEvent] = Field(default_factory=list)
```

**Policy API:**

```python
class EscalationDecision(BaseModel):
    escalate: bool
    reason: str = ""

def record_progress(state: AgentState) -> None: ...
def record_no_progress(state: AgentState, *, kind: str, fingerprint: str, node: str) -> None: ...
def should_escalate(state: AgentState, *, immediate_reason: str = "") -> EscalationDecision: ...
def apply_escalation(state: AgentState, decision: EscalationDecision) -> None: ...
def primary_budget_limit(state: AgentState) -> int: ...
```

The default new-agent total budget becomes 100,000. `primary_budget_limit` is `min(55_000, max(0, state.token_budget - 40_000))`. Crossing it escalates if configured; otherwise existing budget failure behavior applies. Immediate reasons are limited to `empty_completion_after_retries` and `invalid_structured_response_after_retries`. Two consecutive events from the approved no-progress kinds escalate; signature fields use exact canonical fingerprints to detect unchanged plans, contexts, edits, and test failures. `apply_escalation` is idempotent and never downgrades.

- [ ] **Step 1: Add RED policy tests**

Cover each immediate reason, two-event threshold, reset on real progress, unchanged plan/context/test signatures, reserve boundary, disabled/missing escalation config, idempotence, and no-downgrade.

- [ ] **Step 2: Add persistence and payload tests**

Round-trip every new field through `save_run`/`load_run`; assert `agent_payload_from_state` includes safe histories but no config/API key; assert legacy saved states load with defaults.

- [ ] **Step 3: Run the RED tests**

Run: `.venv/bin/python -m pytest tests/test_model_policy.py tests/test_run_store.py tests/test_new_agent.py -q`

Expected: missing state models and policy module.

- [ ] **Step 4: Implement models, policy, initialization, and serialization**

Use stable SHA-256 fingerprints over canonical JSON for comparisons. Append a safe `node_diagnostics` event named `model_escalated` containing only from/to/reason/round. Do not store exception text in `ModelInvocation`; store sanitized exception class only.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_model_policy.py tests/test_run_store.py tests/test_new_agent.py tests/test_budget_scaling.py -q`

Run: `.venv/bin/python -m ruff check src/model_policy.py src/state.py src/new_agent.py src/run_store.py tests/test_model_policy.py tests/test_run_store.py`

Commit: `feat: persist deterministic model escalation state`

---

### Task 4: Implement bounded evidence storage

**Files:**

- Create: `src/evidence.py`
- Modify: `src/state.py`
- Test: `tests/test_evidence.py`
- Test: `tests/test_run_store.py`

**Interfaces (`Evidence` is declared in `src/state.py`; `EvidenceStore` is declared in `src/evidence.py`):**

```python
class Evidence(BaseModel):
    evidence_id: str
    tool: str
    file_path: str | None = None
    symbol: str | None = None
    summary: str
    content: str
    fingerprint: str

class EvidenceAddResult(BaseModel):
    evidence: Evidence
    added: bool

class EvidenceStore:
    def __init__(self, state: AgentState, *, max_items: int = 30, max_content_chars: int = 8_000): ...
    def add(self, *, tool: str, summary: str, content: str, file_path: str | None = None, symbol: str | None = None) -> EvidenceAddResult: ...
    def select(self, evidence_ids: list[str], *, max_total_chars: int = 24_000) -> list[Evidence]: ...
```

Add `evidence: list[Evidence]` to `AgentState`. IDs are deterministic (`ev_` plus a fingerprint prefix), text is secret-redacted and truncated, duplicate fingerprints are not appended, and selection preserves requested order while enforcing the total prompt bound.

- [ ] **Step 1: Add RED tests**

Test stable IDs, duplicate suppression, item/content limits, secret redaction, traversal-safe normalized paths, ordered bounded selection, state/run-store round trip, and legacy state loading.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_evidence.py tests/test_run_store.py -q`

- [ ] **Step 3: Implement `EvidenceStore`**

Reuse one public `redact_secrets(text)` helper from `src/model_provider.py`. Evidence fingerprints include tool, normalized relative path, symbol, and normalized content; they exclude summary wording.

- [ ] **Step 4: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_evidence.py tests/test_run_store.py -q`

Run: `.venv/bin/python -m ruff check src/evidence.py src/state.py tests/test_evidence.py tests/test_run_store.py`

Commit: `feat: add bounded replayable evidence store`

---

### Task 5: Add constrained autonomous tool routing

**Files:**

- Create: `src/tool_policy.py`
- Create: `src/tool_router.py`
- Modify: `src/local_search.py`
- Modify: `src/state.py`
- Test: `tests/test_tool_policy.py`
- Test: `tests/test_tool_router.py`
- Test: `tests/test_local_search.py`

**Schemas and APIs:**

```python
ToolAction = Literal[
    "search_symbol", "search_text", "read_symbol", "read_range",
    "find_references", "list_related_tests", "run_targeted_test",
    "inspect_git_diff", "validate_patch", "request_repair",
    "finish_investigation",
]

class ToolIntent(BaseModel):
    action: ToolAction
    args: dict[str, object] = Field(default_factory=dict)
    reason: str
    expected_evidence: str

class ToolInvocation(BaseModel):
    action: ToolAction
    args_fingerprint: str
    status: Literal["approved", "rejected", "ok", "error", "duplicate"]
    evidence_id: str | None = None
    error_class: str = ""

class ToolPolicy:
    def authorize(self, state: AgentState, intent: ToolIntent, *, calls_this_round: int) -> PolicyDecision: ...

async def route_tool_intent(state: AgentState, intent: ToolIntent, *, calls_this_round: int) -> ToolRouteResult: ...
```

Add `tool_history: list[ToolInvocation]` to `AgentState`. `ToolPolicy` caps eight calls per reasoning round and thirty per sample, rejects duplicate action/args fingerprints, verifies all paths resolve under `state.repo_path`, and only permits `run_targeted_test` arguments that parse to one of: `pytest <repo-relative test selector>`, `python -m pytest <selector>`, or an existing project test command narrowed to a repo-relative selector. It rejects shell metacharacters, environment assignments, pipes, redirects, network commands, absolute paths, and `..`.

`ToolRouter` implements all actions through Python/file APIs or fixed argv subprocess calls. `request_repair` and `finish_investigation` are control results and execute no command. Every successful data tool writes through `EvidenceStore`; a duplicate evidence fingerprint is recorded as no progress.

- [ ] **Step 1: Add RED authorization tests**

Cover every allowed action plus traversal, symlink escape, absolute path, unknown args, arbitrary shell, metacharacters, network-looking command, duplicate call, per-round limit, per-sample limit, and exact repo/ref requirements.

- [ ] **Step 2: Add RED router tests**

Build a tiny temporary Git repository and test symbol/text search, bounded reads, references, related tests, targeted-test fixed argv, diff inspection, patch validation, evidence dedupe, control actions, and sanitized errors.

- [ ] **Step 3: Run RED**

Run: `.venv/bin/python -m pytest tests/test_tool_policy.py tests/test_tool_router.py tests/test_local_search.py -q`

- [ ] **Step 4: Implement policy and router**

Use `Path.resolve()` plus `is_relative_to(repo_root.resolve())`; do not add a general command executor. Normalize args before hashing so key order cannot bypass deduplication. Bound result content before adding evidence.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_tool_policy.py tests/test_tool_router.py tests/test_local_search.py tests/test_tools.py -q`

Run: `.venv/bin/python -m ruff check src/tool_policy.py src/tool_router.py src/local_search.py src/state.py tests/test_tool_policy.py tests/test_tool_router.py`

Commit: `feat: route model tool intents through safety policy`

---

### Task 6: Build an allowlisted escalation packet and model prompts

**Files:**

- Create: `src/escalation.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/reflect.py`
- Test: `tests/test_escalation_packet.py`
- Test: `tests/test_decision_frame.py`

**Interfaces:**

```python
class EscalationPacket(BaseModel):
    issue_title: str
    issue_body: str
    repository: str
    base_commit: str
    evidence: list[Evidence]
    failed_edit_signatures: list[str]
    patch_errors: list[str]
    test_error_summaries: list[str]
    rejected_approaches: list[str]
    required_behavior: str
    remaining_token_budget: int
    remaining_execution_attempts: int

def build_escalation_packet(state: AgentState) -> EscalationPacket: ...
def render_escalation_packet(packet: EscalationPacket) -> str: ...
```

Only named fields are copied. The builder accepts the agent-safe issue seed, never the original SWE-bench evaluation record. It bounds body, evidence, error summaries, and rejected approaches independently; redacts secrets after rendering; and uses `state.repo_ref` as the exact base commit.

- [ ] **Step 1: Add RED allowlist tests**

Populate state/test fixtures with secret sentinels, HTTP payloads, generated filenames, gold patch/test patch/FAIL_TO_PASS/PASS_TO_PASS keys, unrelated conversation, and oversized logs. Assert none appear in model dump or rendered prompt while every required safe field remains.

- [ ] **Step 2: Add prompt-selection tests**

Assert Gemini PLAN/REFLECT retain their bounded current prompt, while escalated state renders only the packet plus the relevant structured schema/tool protocol. Assert the selected `llm_call` provider is `escalation` and the model is the active Opus model.

- [ ] **Step 3: Run RED**

Run: `.venv/bin/python -m pytest tests/test_escalation_packet.py tests/test_decision_frame.py -q`

- [ ] **Step 4: Implement packet construction and prompt branching**

Do not yet execute Opus edits in this task. Wire only deterministic prompt/provider selection and model invocation history. Convert exhausted empty/invalid structured responses into immediate policy events rather than terminal node crashes.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_escalation_packet.py tests/test_decision_frame.py tests/test_llm_json_tolerance.py -q`

Run: `.venv/bin/python -m ruff check src/escalation.py src/nodes/plan.py src/nodes/reflect.py tests/test_escalation_packet.py`

Commit: `feat: send safe bounded context on model escalation`

---

### Task 7: Implement Opus RepairPlan, ContextBuilder, and VerifiedEdit

**Files:**

- Create: `src/repair_flow.py`
- Modify: `src/schemas.py`
- Modify: `src/state.py`
- Modify: `src/nodes/plan.py`
- Test: `tests/test_repair_flow.py`
- Test: `tests/test_decision_schemas.py`
- Test: `tests/test_patch_repair_prompt.py`

**Schemas (declared in `src/state.py` and re-used by `src/schemas.py`, avoiding a `state` ↔ `schemas` import cycle):**

```python
class RepairPlan(BaseModel):
    root_cause: str
    target_files: list[str]
    target_symbols: list[str]
    required_behavior: str
    regression_test_strategy: str
    rejected_approaches: list[str]

class VerifiedEdit(BaseModel):
    file_path: str
    node_target: str | None = None
    search: str
    replace: str
    intent: str

class VerifiedEditBatch(BaseModel):
    edits: list[VerifiedEdit]
```

**Flow API:**

```python
def build_target_context(state: AgentState, plan: RepairPlan) -> list[Evidence]: ...
async def generate_opus_repair(state: AgentState, packet: EscalationPacket) -> tuple[RepairPlan, VerifiedEditBatch]: ...
```

The first Opus call produces only `RepairPlan`. `build_target_context` rejects files outside the checkout/plan, resolves complete Python definitions when possible, and otherwise returns bounded exact text windows. The second call receives only the packet, validated plan, and target evidence IDs/content, and produces `VerifiedEditBatch`. Convert successful edits to existing `PatchEdit` objects only after schema and context validation.

- [ ] **Step 1: Add RED schema/context tests**

Cover valid schemas, empty/duplicate/out-of-scope targets, unique and missing symbols, non-Python bounded fallback windows, intentional new text source files, and context budgets.

- [ ] **Step 2: Add mocked two-call flow tests**

Assert call order, provider/model preservation, no patch fields in the first call, exact target context in the second, and failure before edit generation when targets are invalid.

- [ ] **Step 3: Run RED**

Run: `.venv/bin/python -m pytest tests/test_repair_flow.py tests/test_decision_schemas.py tests/test_patch_repair_prompt.py -q`

- [ ] **Step 4: Implement the two-stage repair flow**

Use existing AST/node-target helpers where possible. Never fall back to an unrestricted fuzzy unified diff. Preserve Gemini's current `PlanDecision` path until escalation.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_repair_flow.py tests/test_decision_schemas.py tests/test_patch_repair_prompt.py tests/test_node_target_converter.py -q`

Run: `.venv/bin/python -m ruff check src/repair_flow.py src/schemas.py src/nodes/plan.py tests/test_repair_flow.py`

Commit: `feat: add two-stage opus verified edit flow`

---

### Task 8: Gate patches and perform two local correction attempts

**Files:**

- Create: `src/patch_gate.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/execute.py`
- Modify: `src/state.py`
- Test: `tests/test_patch_gate.py`
- Test: `tests/test_patch_preflight.py`
- Test: `tests/test_patch_retry.py`

**Interfaces:**

```python
class PatchGateIssue(BaseModel):
    code: Literal[
        "target_missing", "target_ambiguous", "search_missing",
        "scope_violation", "apply_failed", "empty_patch",
        "generated_artifact", "binary_artifact", "oversized_file",
    ]
    file_path: str
    message: str
    correction_context: str = ""

class PatchGateResult(BaseModel):
    accepted: bool
    edits: list[PatchEdit] = Field(default_factory=list)
    issues: list[PatchGateIssue] = Field(default_factory=list)

def validate_patch_batch(state: AgentState, plan: RepairPlan, batch: VerifiedEditBatch) -> PatchGateResult: ...
```

Add `patch_correction_count: int = 0` and `active_repair_plan: RepairPlan | None` to state. The gate allows tracked plan target files or intentional new UTF-8 source/test/docs/config text files. It excludes archives, wheels, virtualenvs, egg metadata, caches, build output, binaries, and oversized files. It requires a unique node target or verbatim search match and simulates all edits in memory before mutating the checkout.

- [ ] **Step 1: Add RED gate tests**

Cover unique node targets, exact searches, ambiguous/missing anchors, target scope, traversal/symlink escape, new text files, empty/no-op edits, generated archives, binaries, oversize, multi-edit atomicity, and substantive diff detection.

- [ ] **Step 2: Add RED correction-loop tests**

Mock Opus to return invalid then valid edits. Assert the same active model receives exact issue code plus a bounded real code window, no REFLECT/test retry is consumed, and the third invalid batch routes to ordinary failure/reflection instead of another local correction.

- [ ] **Step 3: Run RED**

Run: `.venv/bin/python -m pytest tests/test_patch_gate.py tests/test_patch_preflight.py tests/test_patch_retry.py -q`

- [ ] **Step 4: Implement atomic validation and local correction routing**

Reuse `locate_search_block`, node-target conversion, and existing preflight primitives. Keep actual worktree mutation in `execute_fix`; PatchGate returns validated existing `PatchEdit` values. Reset correction count only after an accepted batch.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_patch_gate.py tests/test_patch_preflight.py tests/test_patch_retry.py tests/test_search_hallucination_gate.py tests/test_node_target_apply.py -q`

Run: `.venv/bin/python -m ruff check src/patch_gate.py src/nodes/plan.py src/nodes/execute.py src/state.py tests/test_patch_gate.py`

Commit: `feat: validate and locally repair model edits`

---

### Task 9: Add rolling attempt outcome summaries

**Files:**

- Create: `src/outcome_summary.py`
- Modify: `src/state.py`
- Modify: `src/nodes/reflect.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/run_store.py`
- Test: `tests/test_outcome_summary.py`
- Test: `tests/test_run_store.py`
- Test: `tests/test_decision_frame.py`

**Interfaces:**

- Consumes: `llm_call(..., provider="primary", temperature=0.0)` from Task 2 and the existing `FixAttempt`, `DecisionFrame`, and secret-redaction helpers.
- Produces:

```python
MAX_OUTCOME_SUMMARY_CHARS = 200

class OutcomeSummaryInput(BaseModel):
    previous_summary: str = ""
    plan_signature: str
    patch_outcome: str
    test_failure_class: str
    reflection_action: str

def build_outcome_summary_input(state: AgentState) -> OutcomeSummaryInput: ...
def deterministic_outcome_summary(item: OutcomeSummaryInput) -> str: ...
async def summarize_attempt_outcome(state: AgentState) -> str: ...
```

Add `attempt_outcome_summary: str = ""` and `summary_token_usage: int = 0` to `AgentState`. The async function asks Gemini for JSON `{"summary": "..."}` at temperature zero, sanitizes it, and returns at most 200 Unicode characters. An empty, invalid, or overlong response falls back to `deterministic_outcome_summary`. The end of `reflect_on_failure` stores the replacement summary before routing to PLAN. The PLAN user prompt includes exactly one section:

```text
Completed attempts (rolling summary):
<state.attempt_outcome_summary>
```

It must not append previous summaries or complete conversation history through this section.

- [ ] **Step 1: Write failing summary boundary tests**

Add tests equivalent to:

```python
@pytest.mark.asyncio
async def test_summary_replaces_previous_value_and_is_injected_once(monkeypatch):
    state = make_reflect_state(attempt_outcome_summary="old failed approach")

    async def fake_llm_call(*args, **kwargs):
        assert kwargs["provider"] == "primary"
        assert kwargs["temperature"] == 0.0
        return {"summary": "new factual outcome"}

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)
    result = await summarize_attempt_outcome(state)
    state.attempt_outcome_summary = result
    prompt = build_plan_user_prompt(state)
    assert result == "new factual outcome"
    assert prompt.count("Completed attempts (rolling summary):") == 1
    assert "old failed approach" not in prompt


@pytest.mark.asyncio
async def test_invalid_summary_uses_bounded_deterministic_fallback(monkeypatch):
    state = make_reflect_state()

    async def fake_llm_call(*args, **kwargs):
        return {"summary": "x" * 500}

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)
    summary = await summarize_attempt_outcome(state)
    assert 0 < len(summary) <= 200
    assert "Bearer" not in summary
```

Also test empty/malformed response, secrets/evaluator field redaction, summary-call failure, no summary section for an empty state value, and exact save/load/replay persistence.

- [ ] **Step 2: Run RED tests**

Run: `.venv/bin/python -m pytest tests/test_outcome_summary.py tests/test_run_store.py tests/test_decision_frame.py -q`

Expected: import and state-field failures because the summarizer does not exist.

- [ ] **Step 3: Implement the bounded summarizer**

Build the input only from the latest `FixAttempt`, current/last decision frame, and previous bounded summary. Catch `LLMResponseError`, `ValueError`, `ValidationError`, timeout, and provider errors at this auxiliary boundary. The deterministic fallback format is:

```python
text = (
    f"plan={item.plan_signature}; patch={item.patch_outcome}; "
    f"test={item.test_failure_class}; next={item.reflection_action}"
)
return redact_secrets(text)[:MAX_OUTCOME_SUMMARY_CHARS]
```

Record the summary model call in `model_history`, but debit it from a separate diagnostic counter rather than the Opus-reserved primary budget.

- [ ] **Step 4: Wire REFLECT and PLAN**

Call `summarize_attempt_outcome` after a valid reflection decision is recorded and before `current_phase` becomes PLAN. Extract the PLAN prompt construction into `build_plan_user_prompt(state: AgentState) -> str` if it is currently inline, then add the one conditional summary section shown above.

- [ ] **Step 5: Run GREEN tests and Ruff**

Run: `.venv/bin/python -m pytest tests/test_outcome_summary.py tests/test_run_store.py tests/test_decision_frame.py tests/test_context_loop_brake.py -q`

Run: `.venv/bin/python -m ruff check src/outcome_summary.py src/state.py src/nodes/reflect.py src/nodes/plan.py src/run_store.py tests/test_outcome_summary.py`

- [ ] **Step 6: Commit**

```bash
git add src/outcome_summary.py src/state.py src/nodes/reflect.py src/nodes/plan.py src/run_store.py tests/test_outcome_summary.py tests/test_run_store.py tests/test_decision_frame.py
git commit -m "feat: summarize completed repair attempts"
```

---

### Task 10: Integrate policy, tools, and failure semantics into PLAN/REFLECT

**Files:**

- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/reflect.py`
- Modify: `src/nodes/verify.py`
- Modify: `src/nodes/failure.py`
- Modify: `src/graph.py`
- Modify: `src/new_agent.py`
- Test: `tests/test_model_escalation_integration.py`
- Test: `tests/test_context_loop_brake.py`
- Test: `tests/test_convergence_diversity.py`
- Test: `tests/test_failure_taxonomy.py`

**Runtime protocol:**

1. Before each PLAN/REFLECT LLM call, evaluate the reserve/no-progress policy.
2. Ask the active model for either one `ToolIntent`, a plan/repair result, or a stop decision using a discriminated structured response.
3. Route at most eight tool requests in the current reasoning round and re-prompt with only newly selected evidence IDs.
4. Count progress only when evidence fingerprint, plan signature, edit signature, or test-failure signature changes materially.
5. Gemini follows the existing plan path; escalated Opus follows Tasks 7–8.
6. After two Opus no-progress rounds, require a repair plan or stop with `opus_no_progress_limit`.
7. Infrastructure errors do not increment `retry_count`; syntax/import failures route to direct patch correction; assertion failures route to REFLECT; a repeated unchanged assertion signature requires a different target symbol and then terminates if unchanged again.

- [ ] **Step 1: Add deterministic mocked integration tests**

Cover: Gemini empty completion → Opus repair → accepted edit → passing verification; two Gemini invalid anchors → escalation; tool request → new evidence → plan; duplicate tool request → no-progress; reserve-boundary escalation; no key → compatible Gemini-only failure; Opus no-progress stop; infra error retry accounting; syntax repair routing; repeated assertion termination; and graph replay from an already-escalated saved state.

- [ ] **Step 2: Run RED integration tests**

Run: `.venv/bin/python -m pytest tests/test_model_escalation_integration.py tests/test_context_loop_brake.py tests/test_convergence_diversity.py tests/test_failure_taxonomy.py -q`

- [ ] **Step 3: Wire the runtime protocol**

Keep public entry points and do not rename existing phases. Put reusable loop mechanics in small private helpers rather than duplicating them across PLAN and REFLECT. Record safe model/tool events in state and `node_diagnostics` at each decision. Task 11 adds the new COVERAGE phase after this integration is stable.

- [ ] **Step 4: Run regression-heavy focused verification**

Run: `.venv/bin/python -m pytest tests/test_model_escalation_integration.py tests/test_decision_frame.py tests/test_context_loop_brake.py tests/test_convergence_diversity.py tests/test_patch_retry.py tests/test_new_agent.py tests/test_failure_taxonomy.py -q`

Run: `.venv/bin/python -m ruff check src/nodes/plan.py src/nodes/reflect.py src/nodes/verify.py src/nodes/failure.py src/graph.py src/new_agent.py tests/test_model_escalation_integration.py`

- [ ] **Step 5: Commit**

Commit: `feat: integrate success-first escalation into agent graph`

---

### Task 11: Add dynamic coverage proof and generated-test hard gate

**Files:**

- Create: `src/coverage_gate.py`
- Create: `src/test_generator.py`
- Create: `src/nodes/coverage.py`
- Modify: `src/state.py`
- Modify: `src/graph.py`
- Modify: `src/new_agent.py`
- Modify: `src/nodes/__init__.py`
- Modify: `src/nodes/verify.py`
- Modify: `src/nodes/execute.py`
- Modify: `src/patch_gate.py`
- Test: `tests/test_coverage_gate.py`
- Test: `tests/test_test_generator.py`
- Test: `tests/test_coverage_node.py`
- Test: `tests/test_new_agent.py`

**Interfaces:**

- Consumes: exact checkout fields on `AgentState`, `ToolPolicy`, `VerifiedEditBatch`, `validate_patch_batch`, fixed-argv test execution, `ModelPolicy`, and final worktree diff generation from earlier tasks.
- Produces these persisted fields and phase:

```python
class Phase(str, Enum):
    # existing members remain unchanged
    COVERAGE = "COVERAGE"

CoverageStatus = Literal[
    "pending", "existing_verified", "generated_verified", "failed"
]

# AgentState additions
coverage_status: CoverageStatus = "pending"
coverage_test_files: list[str] = Field(default_factory=list)
coverage_test_command: str = ""
coverage_failure_reason: str = ""
test_generation_attempts: int = 0
coverage_proof: CoverageProof | None = None
```

- Produces these runtime contracts; persisted `TestRunFingerprint` and `CoverageProof` live in `src/state.py`, while orchestration-only models live in `src/coverage_gate.py`:

```python
class ChangedTarget(BaseModel):
    file_path: str
    symbols: list[str] = Field(default_factory=list)

class CoverageCandidate(BaseModel):
    test_files: list[str]
    argv: list[str]
    source: Literal["existing", "generated"]

class TestRunFingerprint(BaseModel):
    exit_code: int
    outcome: Literal["pass", "assertion_failure", "infra"]
    failing_test_ids: list[str] = Field(default_factory=list)
    assertion_fingerprint: str = ""
    summary: str = ""

class CoverageProof(BaseModel):
    source: Literal["existing", "generated"]
    test_files: list[str]
    argv: list[str]
    fixed_runs: list[TestRunFingerprint]
    base_runs: list[TestRunFingerprint]

class CoverageDecision(BaseModel):
    verified: bool
    status: CoverageStatus
    reason: str
    candidate: CoverageCandidate | None = None
    fixed_runs: list[TestRunFingerprint] = Field(default_factory=list)
    base_runs: list[TestRunFingerprint] = Field(default_factory=list)

def collect_changed_targets(repo_path: Path, base_ref: str) -> list[ChangedTarget]: ...
def discover_coverage_candidates(state: AgentState, targets: list[ChangedTarget]) -> list[CoverageCandidate]: ...
def normalize_test_result(exit_code: int, output: str, repo_path: Path) -> TestRunFingerprint: ...
@contextmanager
def temporary_base_checkout(repo_path: Path, base_ref: str) -> Iterator[Path]: ...
async def validate_differential_coverage(state: AgentState, candidate: CoverageCandidate) -> CoverageDecision: ...
def is_allowed_test_path(repo_root: Path, file_path: str) -> bool: ...
async def request_test_batch(state: AgentState, rejection_reason: str) -> VerifiedEditBatch: ...
async def generate_test_batch(state: AgentState, rejection_reason: str) -> VerifiedEditBatch: ...
async def run_test_generation_attempts(state: AgentState, rejection_reason: str) -> CoverageDecision: ...
async def ensure_coverage(state: AgentState | dict[str, Any]) -> AgentState: ...
```

`discover_coverage_candidates` may use symbol references to locate candidate files, but only `validate_differential_coverage` decides coverage. Candidate argv is restricted to the fixed `pytest`/`python -m pytest` forms authorized in Task 5. The validator runs fixed twice, copies the exact candidate test files into a temporary detached checkout at the 40-character `state.repo_ref`, then runs base twice. Verification requires two fixed passes and two base assertion failures with identical failing test IDs and identical normalized assertion fingerprints.

- [ ] **Step 1: Write failing dynamic coverage tests**

Create a temporary Git repository fixture with a base implementation, a fixed implementation, and a targeted test. Cover the accepted and rejected matrices:

```python
@pytest.mark.asyncio
async def test_differential_coverage_requires_stable_fixed_pass_base_failure(
    differential_repo, monkeypatch
):
    state, candidate = differential_repo(
        fixed=[passing_run(), passing_run()],
        base=[assertion_run("test_bug", "expected 2 got 1")] * 2,
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is True
    assert decision.status == "existing_verified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixed", "base", "reason"),
    [
        ([passing_run()] * 2, [passing_run()] * 2, "base_also_passes"),
        ([failing_run()] * 2, [assertion_run()] * 2, "fixed_does_not_pass"),
        ([passing_run()] * 2, [infra_run()] * 2, "coverage_infra"),
        ([passing_run()] * 2, [assertion_run("a"), assertion_run("b")], "unstable_base_failure"),
    ],
)
async def test_differential_coverage_rejects_non_proofs(
    differential_repo, fixed, base, reason
):
    state, candidate = differential_repo(fixed=fixed, base=base)
    result = await validate_differential_coverage(state, candidate)
    assert result.verified is False
    assert result.reason == reason
```

Also test exact base SHA validation, temporary checkout cleanup on success/error, path traversal/symlink rejection, normalized temp-path/timing removal, and fixed argv preservation.

- [ ] **Step 2: Run coverage-gate RED tests**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py -q`

Expected: `src.coverage_gate` does not exist.

- [ ] **Step 3: Implement target collection, candidate discovery, and differential execution**

Use `git diff --unified=0 <base_ref> --` plus AST parsing to derive changed production symbols. Discover tests from approved targeted-test history, `state.test_command`, and repository test files referencing those symbols. Normalize output by replacing repo/temp paths, durations, line numbers, and random IDs before SHA-256 hashing. Treat exit code zero as pass; accept assertion failure only when pytest/unittest reports named failed tests and output contains no collection/import/setup/timeout/crash marker.

Create the base tree with fixed argv equivalent to:

```python
subprocess.run(
    ["git", "worktree", "add", "--detach", str(temp_path), base_ref],
    cwd=repo_path,
    check=True,
    capture_output=True,
    text=True,
)
```

Copy only authorized candidate test files from fixed to base, then always remove the temporary worktree in `finally` with a validated temp path and `git worktree remove --force`.

- [ ] **Step 4: Write failing test-generation policy tests**

Add tests equivalent to:

```python
@pytest.mark.asyncio
async def test_test_generator_is_test_only_and_escalates_second_attempt(monkeypatch):
    state = make_state(active_provider="primary", active_model="gemini-3.5-flash:stable")
    responses = [
        VerifiedEditBatch(edits=[bad_test_edit()]),
        VerifiedEditBatch(edits=[verified_test_edit()]),
    ]
    monkeypatch.setattr("src.test_generator.request_test_batch", async_sequence(responses))
    result = await run_test_generation_attempts(state, "base_also_passes")
    assert result.verified is True
    assert state.test_generation_attempts == 2
    assert state.active_provider == "escalation"
    assert all(is_allowed_test_path(Path(state.repo_path), p) for p in state.coverage_test_files)


@pytest.mark.asyncio
async def test_two_invalid_generated_tests_fail_without_pr(monkeypatch):
    state = make_state()
    monkeypatch.setattr("src.test_generator.request_test_batch", always_invalid_batch)
    result = await ensure_coverage(state)
    assert result.coverage_status == "failed"
    assert result.current_phase == Phase.FAILURE
    assert result.failure_reason.startswith("test_generation_failed:")
    assert result.pr_url is None
```

Cover existing/nonexistent test roots, established `test_*.py` and `*_test.py` layouts, production-file edits, CI/dependency/config edits, generated/binary files, missing escalation credentials, and already-escalated Opus behavior.

- [ ] **Step 5: Run generator RED tests**

Run: `.venv/bin/python -m pytest tests/test_test_generator.py tests/test_coverage_node.py -q`

Expected: generator/node imports and `Phase.COVERAGE` fail.

- [ ] **Step 6: Implement test-only generation and the COVERAGE node**

The generator prompt contains only issue text, `ChangedTarget` values, exact bounded source/test-convention evidence, required fixed behavior, and the sanitized rejection reason. Parse it as `VerifiedEditBatch`, then call:

```python
gate = validate_patch_batch(
    state,
    RepairPlan(
        root_cause="missing behavioral regression test",
        target_files=[edit.file_path for edit in batch.edits],
        target_symbols=[],
        required_behavior="prove the issue fails on base and passes on the fix",
        regression_test_strategy="targeted differential test",
        rejected_approaches=[],
    ),
    batch,
    test_only=True,
)
```

Extend `validate_patch_batch(..., test_only: bool = False)` so test-only mode rejects every non-test path and otherwise retains all normal PatchGate checks. Apply accepted test edits through the existing exact `PatchEdit` executor, validate the affected fixed suite, run differential coverage, and refresh `state.patch_content` from the complete current Git diff so the generated test enters PR/eval output.

On first Gemini generation failure, call `apply_escalation` with reason `test_generation_retry` only when escalation is configured; otherwise retry Gemini. Stop after exactly two attempts.

- [ ] **Step 7: Wire graph routing and eval terminal semantics**

Add `"ensure_coverage": 1200.0` to `PHASE_TIMEOUTS`, register the node, and map `Phase.COVERAGE` to it. Change successful VERIFY to `Phase.COVERAGE` regardless of `skip_commit`. On verified coverage, route to DONE when `skip_commit` is true and COMMIT otherwise. No code path may set DONE directly from VERIFY.

- [ ] **Step 8: Run GREEN and regression tests**

Run: `.venv/bin/python -m pytest tests/test_coverage_gate.py tests/test_test_generator.py tests/test_coverage_node.py tests/test_new_agent.py tests/test_patch_gate.py tests/test_patch_preflight.py tests/test_agent_v2_eval.py -q`

Run: `.venv/bin/python -m ruff check src/coverage_gate.py src/test_generator.py src/nodes/coverage.py src/state.py src/graph.py src/new_agent.py src/nodes/verify.py src/nodes/execute.py src/patch_gate.py tests/test_coverage_gate.py tests/test_test_generator.py tests/test_coverage_node.py`

- [ ] **Step 9: Commit**

```bash
git add src/coverage_gate.py src/test_generator.py src/nodes/coverage.py src/state.py src/graph.py src/new_agent.py src/nodes/__init__.py src/nodes/verify.py src/nodes/execute.py src/patch_gate.py tests/test_coverage_gate.py tests/test_test_generator.py tests/test_coverage_node.py tests/test_new_agent.py
git commit -m "feat: require differential regression coverage"
```

---

### Task 12: Expose safe configuration, replay, and eval reporting

**Files:**

- Modify: `.env.example`
- Modify: `README.md`
- Modify: `eval/agent_v2_harness.py`
- Modify: `eval/report.py`
- Modify: `eval/failure_taxonomy.py`
- Modify: `tests/test_agent_v2_eval.py`
- Modify: `tests/test_eval_report.py`
- Modify: `tests/test_failure_taxonomy.py`
- Modify: `tests/test_documentation_contract.py`

**Configuration contract:**

```text
LLM_MODEL=gemini-3.5-flash:stable
LLM_ESCALATION_MODEL=claude-opus-4-8:stable
LLM_ESCALATION_BASE_URL=https://linoapi.com.cn/v1
LLM_ESCALATION_API_KEY=
REPOPILOT_ESCALATION_ENABLED=0
REPOPILOT_ESCALATION_AFTER_NO_PROGRESS=2
```

The harness enables escalation only when the explicit flag is set and the key exists; it never prints the key. Each result/report adds safe fields: `models_used`, `escalated`, `escalation_reason`, `model_invocations`, `tool_invocations`, `unique_evidence_count`, `max_consecutive_no_progress`, `attempt_outcome_summary`, `coverage_status`, `coverage_test_files`, `coverage_test_command`, `coverage_proof`, `coverage_failure_reason`, and `test_generation_attempts`. Existing result files without them continue to load with empty summary, pending coverage, and no proof. Change `eval.report.main` to `main(argv: list[str] | None = None)` and add `--results-file` and `--summary-file`, defaulting to the current `RESULTS_PATH` and `SUMMARY_PATH`.

- [ ] **Step 1: Add RED eval/report/documentation contracts**

Assert exact defaults, explicit enablement, result fields, per-model aggregates, escalation-reason counts, tool rejection/dedup counts, summary length, coverage-status/proof/test-generation aggregates, legacy compatibility, report CLI path forwarding, and absence of keys/gold/generated zip payloads from predictions and reports. Add a result-contract test proving `success=True` is serialized only when `coverage_status` is `existing_verified` or `generated_verified` and `coverage_proof` is present.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py tests/test_eval_report.py tests/test_failure_taxonomy.py tests/test_documentation_contract.py -q`

- [ ] **Step 3: Implement safe configuration and reporting**

Document secret injection as an environment/runtime-secret-store operation only. Keep `agent_success` distinct from official `resolved`. Add failure labels `opus_no_progress_limit`, `patch_gate_rejected`, `model_gateway_infra`, `coverage_infra`, and `test_generation_failed` without rewriting historical artifacts. Render coverage proof as test files, fixed/base outcomes, failing test IDs, and fingerprints only; do not render full raw test logs.

- [ ] **Step 4: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py tests/test_eval_report.py tests/test_failure_taxonomy.py tests/test_documentation_contract.py tests/test_swe_bench.py -q`

Run: `.venv/bin/python -m ruff check eval/ tests/test_agent_v2_eval.py tests/test_eval_report.py tests/test_failure_taxonomy.py tests/test_documentation_contract.py`

Commit: `docs: expose escalation eval and replay controls`

---

### Task 13: Full deterministic verification and five-sample checkpoint

**Files:**

- Modify: `eval/agent_v2_harness.py`
- Modify: `tests/test_agent_v2_eval.py`
- Verify: `src/`, `tests/`, `eval/`
- Generate ignored input: `eval/checkpoint_5_ids.txt`
- Generate ignored artifacts: `eval/checkpoint_5.json`, `eval/checkpoint_5_summary.md`, `eval/checkpoint_5_predictions.jsonl`

- [ ] **Step 1: Add deterministic instance selection and result-path CLI options**

Add `sample_ids: list[str] | None = None` to `run_agent_v2_eval`. For `swe-bench-verified`, load the pinned cached dataset then select those IDs in caller-supplied order; reject missing or duplicate IDs. Add `--sample-ids-file` (one nonblank instance ID per line) and `--results-file` CLI arguments and forward them to `run_agent_v2_eval`. Add tests for ordered selection, unknown/duplicate rejection, results-path forwarding, and unchanged seed-based selection when the option is absent.

- [ ] **Step 2: Run the CLI RED test, implement, verify, and commit**

Run before implementation: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py -q`

Expected: the new option/selection tests fail because neither CLI option exists.

Run after implementation: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py tests/test_swe_bench.py -q`

Run: `.venv/bin/python -m ruff check eval/agent_v2_harness.py tests/test_agent_v2_eval.py`

Commit: `feat(eval): select pinned swe-bench instances`

- [ ] **Step 3: Run all unit/integration checks**

Run: `.venv/bin/python -m pytest -q`

Expected: every existing and new test passes; the previously established skipped test remains the only skip unless its dependency is now available.

Run: `.venv/bin/python -m ruff check src/ tests/ eval/`

Run: `.venv/bin/python -m build`

- [ ] **Step 4: Run secret/gold/generated-artifact scans before live calls**

Run: `rg -n 'sk-[A-Za-z0-9]|Bearer [A-Za-z0-9]|FAIL_TO_PASS|PASS_TO_PASS|gold_patch|test_patch' src tests eval README.md .env.example docs/superpowers/plans/2026-07-20-success-first-model-escalation-tools.md`

Expected: only deliberate test fixture names/assertions and design documentation references appear; no concrete credential or gold content appears.

- [ ] **Step 5: Create the exact five-instance checkpoint input**

Create ignored `eval/checkpoint_5_ids.txt` with this exact order:

```text
pytest-dev__pytest-10081
pydata__xarray-7233
django__django-11815
sympy__sympy-24562
mwaskom__seaborn-3069
```

- [ ] **Step 6: Run the five-sample SWE-bench Verified checkpoint**

With credentials supplied only through the runtime environment, run the same fixed dataset revision/sample IDs as the prior ten-sample baseline, selecting the previous success control plus the xarray, Django, SymPy, and seaborn failures:

```bash
REPOPILOT_ESCALATION_ENABLED=1 \
LLM_MODEL=gemini-3.5-flash:stable \
LLM_ESCALATION_MODEL=claude-opus-4-8:stable \
.venv/bin/python -u eval/agent_v2_harness.py \
  --dataset swe-bench-verified \
  --sample-ids-file eval/checkpoint_5_ids.txt \
  --max-retries 3 \
  --token-budget 100000 \
  --results-file eval/checkpoint_5.json \
  --predictions-file eval/checkpoint_5_predictions.jsonl
```

- [ ] **Step 7: Inspect checkpoint acceptance signals**

Run:

```bash
.venv/bin/python eval/report.py \
  --results-file eval/checkpoint_5.json \
  --summary-file eval/checkpoint_5_summary.md
```

Generate the report and assert: any counted success has `coverage_status` equal to `existing_verified` or `generated_verified` plus a nonempty differential proof; at least one former `search_not_found` sample reaches an applicable patch; empty Gemini output escalates; no sample exceeds two consecutive no-progress PLAN/REFLECT rounds; every completed REFLECT leaves a summary of at most 200 characters; predictions use only `instance_id`, `model_name_or_path`, and `model_patch`; and locally generated tests appear only after differential validation.

- [ ] **Step 8: Commit any checkpoint-driven code fix separately**

If a deterministic product bug is found, reproduce it with a failing test, fix it, rerun Steps 3–7, and use a narrowly named commit. Do not tune against gold patches or evaluator-only fields.

---

### Task 14: Ten-sample acceptance run and official scoring boundary

**Files:**

- Generate ignored input: `eval/baseline_10_ids.txt`
- Generate ignored artifacts: `eval/success_first_10.json`, `eval/success_first_10_summary.md`, `eval/swe_bench_predictions.jsonl`
- Modify only if results reveal a test-reproduced product defect: corresponding `src/`, `tests/`, or `eval/` files

- [ ] **Step 1: Create the exact baseline instance input**

Create ignored `eval/baseline_10_ids.txt` with this exact order, matching the 1/10 baseline:

```text
pydata__xarray-7233
astropy__astropy-7336
scikit-learn__scikit-learn-12682
pytest-dev__pytest-10081
django__django-11815
mwaskom__seaborn-3069
psf__requests-1724
pallets__flask-5014
sympy__sympy-24562
pylint-dev__pylint-6903
```

- [ ] **Step 2: Run the deterministic ten-sample evaluation**

Run the same pinned SWE-bench Verified revision and exact ten instance IDs as the 1/10 baseline with escalation enabled, `max_retries=3`, and `token_budget=100000`:

```bash
REPOPILOT_ESCALATION_ENABLED=1 \
LLM_MODEL=gemini-3.5-flash:stable \
LLM_ESCALATION_MODEL=claude-opus-4-8:stable \
.venv/bin/python -u eval/agent_v2_harness.py \
  --dataset swe-bench-verified \
  --sample-ids-file eval/baseline_10_ids.txt \
  --max-retries 3 \
  --token-budget 100000 \
  --results-file eval/success_first_10.json \
  --predictions-file eval/swe_bench_predictions.jsonl
```

- [ ] **Step 3: Verify internal acceptance**

Run:

```bash
.venv/bin/python eval/report.py \
  --results-file eval/success_first_10.json \
  --summary-file eval/success_first_10_summary.md
```

Assert at least 4/10 `agent_success`, with every success carrying a stable two-pass fixed/base `coverage_proof`; at least two of the four former `search_not_found` instances recovered; empty-completion escalation; summaries bounded to 200 characters; safe replay fields; no more than two consecutive no-progress rounds; and no credentials/gold/generated artifacts in outputs. Confirm two failed test-generation attempts are classified `test_generation_failed` and never counted as success. Report the actual outcome without relabeling failures.

- [ ] **Step 4: Run official SWE-bench scoring only when Docker is available**

Run:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --predictions_path eval/swe_bench_predictions.jsonl \
  --max_workers 1 \
  --run_id repopilot-success-first-20260720
```

If Docker/image infrastructure is unavailable, record official scoring as blocked by infrastructure and retain the prediction JSONL; never infer `resolved` from RepoPilot's internal test outcome.

- [ ] **Step 5: Perform final verification and branch review**

Run: `.venv/bin/python -m pytest -q`

Run: `.venv/bin/python -m ruff check src/ tests/ eval/`

Run: `.venv/bin/python -m build`

Run: `git status --short && git log --oneline --decorate -12`

Use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`, and finally `superpowers:finishing-a-development-branch`. Push only committed source/docs/tests; keep credentials and ignored live artifacts out of Git.
