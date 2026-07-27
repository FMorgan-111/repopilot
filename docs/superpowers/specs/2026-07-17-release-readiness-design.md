# RepoPilot Release Readiness Design

## Objective

Make the `feat/caching-convergence-xrepo-memory` branch installable, testable,
portable, provider-neutral, and honestly measurable before it is merged into
`master`.

## Scope

This change covers five connected release-readiness contracts:

1. Packaging and CI install the dependencies required by the shipped runtime
   and test suite.
2. Cross-repository episode memory runs on Python builds that support
   `sqlite-vec` and falls back to a deterministic NumPy index when SQLite
   extension loading is unavailable.
3. The LLM client has provider-neutral configuration and accepts both streamed
   SSE and ordinary JSON chat-completion responses without silently accepting
   empty or malformed responses.
4. Evaluation reports distinguish oracle-file (`--seed-gold-files`) results
   from end-to-end results.
5. User-facing and contributor documentation describes the behavior that the
   code and CI actually provide.

Large-scale refactoring of the agent state machine and improvements to patch
generation quality are explicitly out of scope. Those should follow after the
release contracts are stable.

## Packaging and CI

`pyproject.toml` is the source of truth for installation metadata. Runtime
dependencies needed by every installation remain in `project.dependencies`.
The optional `memory` extra contains `fastembed`, `numpy`, and `sqlite-vec`.
The `dev` extra contains pytest, pytest-asyncio, ruff, and the memory extra's
test-time dependencies.

CI installs `.[memory,dev]`, runs the complete test suite, and runs Ruff. The
test matrix includes Linux and macOS so the SQLite extension-loading difference
is continuously exercised. `requirements.txt` remains a pinned convenience
file but must contain the same runtime and memory dependency set.

## Portable Vector Index

`ErrorEpisodeStore` receives its vector index from a small backend factory. The
factory prefers `SqliteVecIndex` when the current Python SQLite connection can
load extensions. If extension loading is missing or `sqlite-vec` cannot be
loaded, it creates a `NumpyVectorIndex` backed by a normal SQLite table.

Both backends implement the existing `VectorIndex` protocol and use cosine
distance with the same ordering semantics. A caller can inspect the selected
backend through a stable diagnostic property. Fallback is visible through a
warning, not silent.

## LLM Configuration and Response Handling

Configuration stays compatible with the existing environment variables:
`LLM_API_KEY`, `DEEPSEEK_API_KEY`, `LINOAPI_API_KEY`, `OPENAI_BASE_URL`, and
`LLM_MODEL`. The project defaults to the user-selected LinoAPI configuration:
`https://linoapi.com.cn/v1` and `claude-sonnet-5:stable`. Other
OpenAI-compatible gateways and models remain selectable through environment
variables.

The HTTP client requests streaming by default. Its response adapter:

- parses `text/event-stream` chunks and ordinary JSON responses;
- accumulates content, finish reason, usage, and tool-call deltas when present;
- raises a typed response error for explicit SSE errors, malformed JSON,
  unsupported response shapes, or a completed response with neither content
  nor tool calls;
- preserves the existing wall-clock timeout and retry policy.

The public return value remains OpenAI-chat-completion-shaped so downstream
nodes do not need to change.

## Evaluation Semantics

Every evaluation result records an `evaluation_mode` field with one of:
`end_to_end` or `oracle_files`. Aggregate reports group resolved rate and
failure taxonomy by this field. `--seed-gold-files` always means
`oracle_files`; an unseeded run always means `end_to_end`.

Existing result files without the field are treated as `end_to_end` for
backward compatibility. Reports must print the mode next to sample count,
model, commit SHA when available, and resolved rate.

## Documentation

README installation commands use the declared extras, explain that episode
memory is opt-in and may download an embedding model, and describe the neutral
LLM configuration. Progress documents are marked as historical snapshots and
must not claim that deleted generated artifacts are present in Git.

## Testing and Acceptance

Acceptance requires all of the following:

- A clean editable install with `.[memory,dev]` succeeds.
- The full test suite passes on the current macOS environment.
- Tests prove the NumPy fallback produces correctly ordered cosine-nearest
  results and is selected when extension loading is unavailable.
- Tests cover SSE success, JSON success, usage and finish-reason preservation,
  explicit error events, malformed/empty streams, and tool-call-only responses.
- Evaluation tests prove seeded and unseeded modes are labeled separately.
- Ruff passes for `src`, `tests`, and the modified evaluation modules.
- README and contributor commands match the package metadata and CI commands.
