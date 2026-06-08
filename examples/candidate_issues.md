# Candidate Issues for Demo

## Case 1: Type conversion bug in CLI boolean overrides

- URL: https://github.com/cookiecutter/cookiecutter/issues/1973
- Why suitable: Simple type conversion bug — CLI overrides are strings but boolean config vars expect booleans. The fix is a straightforward isinstance check and conversion. Small code change, clear expected behavior.
- Expected fix: In `cookiecutter/generate.py`, when applying CLI overrides, check if the target config variable is a bool and convert the string override (e.g., "true"/"false"/"yes"/"no") to a boolean before applying.
- Repo: cookiecutter/cookiecutter (Python, ~25k stars, moderate size)

## Case 2: Missing None check in import path

- URL: https://github.com/Textualize/textual/issues/3996
- Why suitable: Simple race-condition bug — `os.stat()` can raise `FileNotFoundError` when a file is temporarily deleted (e.g., by vim during save). The fix is adding error handling around the stat call. Clear root cause identified in the issue body.
- Expected fix: In `src/textual/file_monitor.py`, wrap the `os.stat()` call in a try/except `FileNotFoundError` block, or add an existence check before calling stat.
- Repo: Textualize/textual (Python, ~36k stars, TUI framework)

## Case 3: Wrong variable name / type hint mismatch

- URL: https://github.com/tiangolo/fastapi/issues/368
- Why suitable: A straightforward type hint bug — `field.validate()` expects a Dict but receives a `Response` object. The fix is renaming the parameter from `response` to `response_content` and adjusting the call site. The issue points directly to the problematic line of code.
- Labels: bug, good first issue, confirmed, reviewed
- Expected fix: In `fastapi/routing.py`, rename the parameter `response: Response` to `response_content: Any` in `serialize_response()`, and update the call sites to pass the response body/content instead of the Response object.
- Repo: tiangolo/fastapi (Python, ~99k stars, web framework)

---

## Notes

- These issues are all from repos larger than the ideal <1000 star target, but they are the most suitable candidates found in the existing dataset at `/mnt/e/hermes-work/repopilot/data/`.
- Each has: a clear bug description, an identified root cause (specific file/module mentioned), and a small, focused fix.
- The RepoPilot CLI could not be run due to network restrictions in this session. To run these:
  ```bash
  cd /mnt/e/hermes-work/repopilot
  python3 -m src.cli <issue_url> --dry-run --token-budget 30000 --json > examples/case_<N>.json 2>&1
  ```
