# Real Demo Cases for RepoPilot

## How to run

```bash
cd /mnt/e/hermes-work/repopilot
repopilot <issue_url> --dry-run --json > examples/traces/case_N.json 2>&1
```

## Candidate Issues (manually curated)

### Case 1: python-dotenv — CLI override boolean bug
- URL: https://github.com/theskumar/python-dotenv/issues/525
- Stars: ~7k (moderate)
- Bug: `dotenv set` command doesn't handle boolean values correctly
- Expected fix: 1 file change in cli.py
- Why RepoPilot-suitable: Small codebase (~20 files), simple CLI logic, clear bug description

### Case 2: pytest-httpserver — Error handling in request matching
- URL: https://github.com/csernazs/pytest-httpserver/issues/345
- Stars: ~200 (small!)
- Bug: Request matching fails silently when headers mismatch
- Expected fix: 1-2 files, add proper error logging
- Why RepoPilot-suitable: Very small repo, simple logic, clear fix direction

### Case 3: typer — Missing type coercion for List types
- URL: https://github.com/fastapi/typer/issues/931
- Stars: ~17k (moderate)
- Bug: CLI List parameters not properly coerced from string input
- Expected fix: 1-2 files in the parser module
- Why RepoPilot-suitable: Focused fix, well-structured codebase
