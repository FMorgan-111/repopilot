# Gemini Default Model Design

**Date:** 2026-07-20

## Goal

Change RepoPilot's default OpenAI-compatible chat-completion model from
`claude-sonnet-5:stable` and remaining legacy DeepSeek defaults to
`gemini-3.5-flash:stable`, while retaining `LLM_MODEL` as the runtime override.
After verification, run ten Agent V2 end-to-end evaluation samples with the
Gemini default.

## Approaches Considered

1. **Environment override only.** Run eval with `LLM_MODEL` set to Gemini but
   leave repository defaults unchanged. This is fastest, but later commands can
   silently return to Claude or DeepSeek.
2. **Runtime default only.** Change `src/http_client.py` and leave eval and
   documentation defaults unchanged. This creates inconsistent behavior across
   entry points.
3. **Unified defaults (selected).** Update runtime, legacy eval, documentation,
   example configuration, and contract tests together. This makes the model
   choice predictable while preserving explicit overrides.

## Design

### Configuration contract

- `LLM_MODEL` remains the highest-priority user-configurable model setting.
- When `LLM_MODEL` is absent, production Agent V2 and legacy evaluation code
  use `gemini-3.5-flash:stable`.
- `OPENAI_BASE_URL` continues to default to `https://linoapi.com.cn/v1` in the
  production HTTP client.
- API credentials remain environment-only and must not be written to tracked
  files, logs, documentation, or generated reports.

### Files in scope

- `src/http_client.py`: Agent V2 runtime default.
- `eval/harness.py`: all legacy eval function, CLI, and module defaults.
- `README.md` and `.env.example`: documented configuration examples.
- Contract/unit tests that assert the default model or render model names.

Historical eval result files are not rewritten to claim they used Gemini.

### Evaluation flow

After the configuration change passes tests, execute:

```text
python -u eval/harness.py --agent-v2 --samples 10 --max-retries 3 --token-budget 50000
```

The process receives `LLM_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`, and
`GITHUB_TOKEN` through environment variables. The run is unseeded, so every
result is labeled `end_to_end`. Results are written to
`eval/eval_results.json`, and the report generator produces the corresponding
summary.

### Failure handling

- Preserve existing HTTP retry and response-normalization behavior.
- Record per-sample failures instead of reporting them as successes.
- If the environment blocks RepoPilot's state directory, grant access to the
  dedicated `~/.repopilot` directory rather than changing `HOME`.
- Never retry the entire batch merely to conceal real model or network
  failures; rerun only when the batch itself is invalidated by infrastructure
  setup.

## Verification

1. Add or update tests so an unset `LLM_MODEL` resolves to
   `gemini-3.5-flash:stable`.
2. Assert documentation and example configuration advertise the Gemini model.
3. Assert no stale hard-coded Claude or DeepSeek default remains in production
   or eval entry points.
4. Run focused model/configuration tests, then the full test suite and Ruff.
5. Run ten live Agent V2 end-to-end eval samples and generate the eval report.
6. Confirm all generated results identify `gemini-3.5-flash:stable` and contain
   no credential material.

## Out of Scope

- Changing the LinoAPI endpoint or authentication scheme.
- Adding provider-specific Gemini SDK code.
- Altering token budgets, retry counts, prompts, or agent graph behavior.
- Reclassifying historical Claude evaluation results as Gemini results.
