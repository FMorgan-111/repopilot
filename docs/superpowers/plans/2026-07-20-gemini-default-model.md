# Gemini Default Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gemini-3.5-flash:stable` the consistent RepoPilot default model and run ten Agent V2 end-to-end evaluations with that model.

**Architecture:** Keep the existing OpenAI-compatible HTTP path and environment-variable override mechanism. Replace divergent hard-coded defaults with one Gemini default per entry point, align documentation and tests, then run the unchanged Agent V2 evaluation workflow.

**Tech Stack:** Python 3.11, pytest, Ruff, httpx, LinoAPI OpenAI-compatible chat completions, GitHub CLI.

## Global Constraints

- The exact default model is `gemini-3.5-flash:stable`.
- `LLM_MODEL` remains the runtime override.
- The default API base remains `https://linoapi.com.cn/v1`.
- API keys and GitHub tokens remain environment-only and must not be written to tracked files, logs, documentation, or reports.
- Live evaluation remains unseeded `end_to_end`, with 10 samples, 3 retries, and a 50,000-token budget per sample.
- Do not rewrite historical Claude results to claim they used Gemini.

---

### Task 1: Production runtime default

**Files:**
- Modify: `tests/test_http_client.py`
- Modify: `src/http_client.py`

**Interfaces:**
- Consumes: environment variable `LLM_MODEL: str | None`.
- Produces: `src.http_client._get_llm_model() -> str`, returning the environment value when set and `gemini-3.5-flash:stable` otherwise.

- [ ] **Step 1: Change the runtime contract test first**

Replace the existing Claude default test with:

```python
def test_default_llm_model_is_gemini_flash(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert _get_llm_model() == "gemini-3.5-flash:stable"
```

- [ ] **Step 2: Run the focused test and verify the red state**

Run: `.venv/bin/python -m pytest tests/test_http_client.py::test_default_llm_model_is_gemini_flash -q`

Expected: one failure showing the actual value is `claude-sonnet-5:stable`.

- [ ] **Step 3: Change the production fallback**

Update `src/http_client.py`:

```python
def _get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "gemini-3.5-flash:stable")
```

- [ ] **Step 4: Run the focused test and HTTP client suite**

Run: `.venv/bin/python -m pytest tests/test_http_client.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the runtime change**

```bash
git add src/http_client.py tests/test_http_client.py
git commit -m "feat: default runtime model to Gemini Flash"
```

---

### Task 2: Legacy eval and documented defaults

**Files:**
- Modify: `tests/test_agent_v2_eval.py`
- Modify: `tests/test_documentation_contract.py`
- Modify: `eval/harness.py`
- Modify: `README.md`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `LLM_MODEL` and the existing `--model` CLI argument.
- Produces: `eval.harness.DEFAULT_MODEL: str` with value `gemini-3.5-flash:stable`; every legacy eval default and CLI fallback refers to that constant.

- [ ] **Step 1: Add failing eval and documentation contracts**

Add to `tests/test_agent_v2_eval.py`:

```python
def test_legacy_harness_default_model_is_gemini_flash():
    from eval import harness

    assert harness.DEFAULT_MODEL == "gemini-3.5-flash:stable"
```

Update the README assertion and add `.env.example` checks in `tests/test_documentation_contract.py`:

```python
def test_readme_documents_installation_memory_and_selected_model():
    readme = _read("README.md")

    assert 'pip install "repopilot[memory]"' in readme
    assert 'pip install -e ".[memory,dev]"' in readme
    assert "REPOPILOT_ENABLE_EPISODES=1" in readme
    assert "gemini-3.5-flash:stable" in readme
    assert "NumPy" in readme


def test_env_example_uses_openai_compatible_gemini_defaults():
    example = _read(".env.example")

    assert "LLM_API_KEY=***" in example
    assert "OPENAI_BASE_URL=https://linoapi.com.cn/v1" in example
    assert "LLM_MODEL=gemini-3.5-flash:stable" in example
```

- [ ] **Step 2: Run the new contracts and verify the red state**

Run: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py::test_legacy_harness_default_model_is_gemini_flash tests/test_documentation_contract.py -q`

Expected: failures for missing `DEFAULT_MODEL` and stale Claude/DeepSeek examples.

- [ ] **Step 3: Introduce one legacy eval constant**

At the eval configuration boundary in `eval/harness.py`, define and reuse:

```python
DEFAULT_MODEL = "gemini-3.5-flash:stable"
_LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL)
```

Replace every `model: str = "deepseek-v4-flash"` default and the CLI declaration with `DEFAULT_MODEL`, for example:

```python
async def llm_call_structured(
    system: str, user: str, model: str = DEFAULT_MODEL
) -> tuple[dict, int, int]:
    ...


parser.add_argument("--model", default=DEFAULT_MODEL)
```

Apply the same replacement to `_llm_search_context`, `evaluate_sample`, and `run_eval`. Keep the historical `DEEPSEEK_PRICING` cost table unchanged because provider pricing is outside this change.

- [ ] **Step 4: Align README and example environment configuration**

Use this README model line:

```bash
export LLM_MODEL=gemini-3.5-flash:stable  # optional; this is the default
```

Use these provider-neutral `.env.example` entries:

```dotenv
# OpenAI-compatible API key
LLM_API_KEY=***

# OpenAI-compatible endpoint
OPENAI_BASE_URL=https://linoapi.com.cn/v1

# Optional: custom model
LLM_MODEL=gemini-3.5-flash:stable
```

- [ ] **Step 5: Run focused eval and documentation tests**

Run: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py tests/test_documentation_contract.py tests/test_http_client.py -q`

Expected: all tests pass.

- [ ] **Step 6: Scan active entry points for stale defaults**

Run: `rg -n 'claude-sonnet-5:stable|model: str = "deepseek-v4-flash"|default="deepseek-v4-flash"' src eval README.md .env.example tests`

Expected: no active default or contract assertions for Claude/DeepSeek; historical fixtures may remain only where they deliberately describe historical results.

- [ ] **Step 7: Commit the eval and documentation change**

```bash
git add eval/harness.py README.md .env.example tests/test_agent_v2_eval.py tests/test_documentation_contract.py
git commit -m "feat: use Gemini Flash across model defaults"
```

---

### Task 3: Repository verification

**Files:**
- Verify: `src/`
- Verify: `eval/`
- Verify: `tests/`

**Interfaces:**
- Consumes: Tasks 1 and 2 commits.
- Produces: test, lint, and configuration-scan evidence for the Gemini switch.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: zero failures.

- [ ] **Step 2: Run Ruff**

Run: `.venv/bin/python -m ruff check src/ tests/ eval/`

Expected: `All checks passed!`

- [ ] **Step 3: Verify the resolved defaults without revealing secrets**

Run: `.venv/bin/python -c 'from src.http_client import _get_llm_model; from eval.harness import DEFAULT_MODEL; print(_get_llm_model()); print(DEFAULT_MODEL)'`

Expected output:

```text
gemini-3.5-flash:stable
gemini-3.5-flash:stable
```

- [ ] **Step 4: Verify no credential material was introduced**

Run: `git diff HEAD~2 -- . ':!docs/superpowers/specs/*' ':!docs/superpowers/plans/*'`

Expected: no API key or GitHub token value; only model/configuration, documentation, and test changes.

---

### Task 4: Ten-sample Gemini live evaluation

**Files:**
- Generate: `eval/eval_results.json` (ignored runtime artifact)
- Generate: `eval/eval_summary.md` (ignored runtime artifact)
- Generate: `examples/traces/trace_*.json` (ignored runtime artifacts)

**Interfaces:**
- Consumes: verified Gemini defaults, `LLM_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`, `GITHUB_TOKEN`, and the first ten records in `data/samples/issues_fixes.jsonl`.
- Produces: ten `end_to_end` result records and a Markdown aggregate report.

- [ ] **Step 1: Confirm required credential names are available without printing values**

Start the process from a shell where the user-provided credential is already
exported as `LLM_API_KEY`, then run `gh auth status --hostname github.com`.
Do not print either token.

- [ ] **Step 2: Run the ten-sample batch**

Run with environment variables supplied only to the process:

```bash
OPENAI_BASE_URL=https://linoapi.com.cn/v1 \
LLM_MODEL=gemini-3.5-flash:stable \
GITHUB_TOKEN="$(gh auth token)" \
.venv/bin/python -u eval/harness.py \
  --agent-v2 --samples 10 --max-retries 3 --token-budget 50000
```

Expected: the command reaches `Agent v2 eval results saved to .../eval/eval_results.json`; individual samples may pass or fail and must retain their real outcomes.

- [ ] **Step 3: Generate the aggregate report**

Run: `.venv/bin/python eval/report.py`

Expected: `eval/eval_summary.md` is generated from the new results.

- [ ] **Step 4: Validate batch identity and count**

Run:

```bash
.venv/bin/python -c 'import json; rows=json.load(open("eval/eval_results.json")); assert len(rows)==10; assert {r["model"] for r in rows}=={"gemini-3.5-flash:stable"}; assert {r["evaluation_mode"] for r in rows}=={"end_to_end"}; print(len(rows))'
```

Expected output: `10`.

- [ ] **Step 5: Scan generated artifacts for credential names and token prefixes**

Run: `rg -n 'Authorization|Bearer |sk-[A-Za-z0-9]|gho_|ghp_|github_pat_' eval/eval_results.json eval/eval_summary.md examples/traces/trace_*.json`

Expected: no matches.

- [ ] **Step 6: Summarize actual results**

Report total samples, successes, failures, success rate, median/total elapsed time when available, token usage, and the dominant failure categories. Keep infrastructure failures separate from model/patch/test failures.

---

### Task 5: Publish the branch update

**Files:**
- Verify: Git history and branch tracking only.

**Interfaces:**
- Consumes: committed source/docs/test changes and verified live eval evidence.
- Produces: updated remote branch `fix/release-readiness-20260717`.

- [ ] **Step 1: Confirm only intended tracked changes remain**

Run: `git status --short --branch`

Expected: clean tracked worktree on `fix/release-readiness-20260717`; ignored eval artifacts do not appear.

- [ ] **Step 2: Push the existing feature branch**

Run: `git push origin fix/release-readiness-20260717`

Expected: remote branch advances to the latest local commit.

- [ ] **Step 3: Verify local and remote commit identity**

Run: `git rev-parse HEAD`

Run: `git ls-remote --heads origin refs/heads/fix/release-readiness-20260717`

Expected: both commands return the same commit SHA.
