"""Benchmark: how often does Pydantic validation pass on first try with deepseek-v4-pro?

Runs classify_issue (20x), rank_files (20x), generate_fix_plan (20x).
Reports: first-try pass rate, retry success rate, fallback rate.
"""

import asyncio
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Actual LLM calls, no mocking
from src.llm import llm_call, _config, _extract_json
from src.schemas import Classification, FileRanking, FixPlan
from pydantic import ValidationError

# Test data — varied issue titles/descriptions
CLASSIFY_INPUTS = [
    ("Login button not working on mobile", "The login button on mobile devices does not respond to taps. Repro steps: 1. Open app 2. Tap login 3. Nothing happens."),
    ("Add dark mode support", "Users want dark mode for better readability at night. Should respect system preference and have manual toggle."),
    ("Memory leak in websocket handler", "After 48h uptime, memory grows from 200MB to 3GB. Heap dump shows unclosed WebSocket connections accumulating."),
    ("Update README with API docs link", "The README is missing a link to the new API documentation at /docs/api-v2."),
    ("Race condition in payment processing", "Two concurrent requests can double-charge users. Happens when user clicks 'pay' twice before first request completes."),
    ("SQL injection in search endpoint", "User input in /api/search?q= is not parameterized. Can dump entire user table."),
    ("Add export to CSV feature", "Admin dashboard should have an export button that downloads filtered results as CSV."),
    ("Timeout on image upload for large files", "Uploading images > 20MB times out after 30s. Nginx and app timeout settings both need adjustment."),
    ("Fix typos in error messages", "Several error messages have spelling mistakes: 'occured' -> 'occurred', 'recieved' -> 'received'."),
    ("Performance degradation on list endpoint", "GET /api/tasks slowed from 200ms to 3s after last deploy. Likely missing database index."),
    ("Broken pagination on search results", "Page 2+ returns empty results even when there are more than 20 items."),
    ("Add rate limiting to auth endpoints", "Currently no rate limiting on /login and /register — vulnerable to brute force attacks."),
    ("Localization: add Chinese translations", "All UI strings are hardcoded in English. Need i18n support with at least Chinese."),
    ("Webhook delivery retry logic broken", "Failed webhooks are retried exactly once instead of exponential backoff. Should follow Stripe-style retry schedule."),
    ("Upgrade Node.js from 18 to 22", "CI still uses Node 18 which is EOL. Need to update .nvmrc, Dockerfile, and CI config."),
    ("API returns 500 on empty request body", "POST /api/submit with empty body crashes instead of returning 400."),
    ("Add health check endpoint", "Need /health endpoint for K8s liveness probe — should check DB and Redis connectivity."),
    ("Session cookie not marked HttpOnly", "Auth cookie missing HttpOnly flag — XSS can steal session tokens."),
    ("Dependency vulnerability in lodash", "npm audit reports CVE-2024-xxxx in lodash 4.17.20. Need to upgrade to 4.17.21+."),
    ("Feature flag cleanup — remove old experiments", "Three feature flags from Q1 experiments are always-on now. Remove dead code paths."),
]

RANK_FILES = [
    # (issue_title, issue_body, file_list)
    ("Login button not working", "Tap login on mobile does nothing.",
     [{"path": "src/components/LoginButton.tsx"}, {"path": "src/hooks/useAuth.ts"},
      {"path": "src/api/auth.ts"}, {"path": "tests/login.test.tsx"},
      {"path": "src/styles/login.css"}]),
    ("Memory leak in websocket", "Memory grows over time from unclosed connections.",
     [{"path": "src/ws/handler.py"}, {"path": "src/ws/connection.py"},
      {"path": "src/app.py"}, {"path": "tests/ws_test.py"},
      {"path": "requirements.txt"}]),
    ("SQL injection in search", "User input not parameterized.",
     [{"path": "src/api/search.py"}, {"path": "src/db/query_builder.py"},
      {"path": "src/middleware/input_validation.py"}, {"path": "tests/api/test_search.py"},
      {"path": "README.md"}]),
    ("Add dark mode support", "Users want dark mode toggle.",
     [{"path": "src/theme/ThemeProvider.tsx"}, {"path": "src/styles/variables.css"},
      {"path": "src/components/Header.tsx"}, {"path": "src/components/Settings.tsx"},
      {"path": "src/hooks/usePreferences.ts"}]),
    ("Performance degradation on list", "Endpoint slowed from 200ms to 3s.",
     [{"path": "src/api/tasks.py"}, {"path": "src/db/models.py"},
      {"path": "src/db/indexes.sql"}, {"path": "src/cache/redis.py"},
      {"path": "config/nginx.conf"}]),
    ("Broken pagination", "Page 2 returns empty.",
     [{"path": "src/api/pagination.py"}, {"path": "src/api/search.py"},
      {"path": "src/db/queries.py"}, {"path": "frontend/src/Paginator.tsx"},
      {"path": "tests/pagination.test.py"}]),
    ("Race condition payment", "Double charge on concurrent clicks.",
     [{"path": "src/payment/processor.py"}, {"path": "src/payment/idempotency.py"},
      {"path": "src/api/checkout.py"}, {"path": "src/db/transactions.py"},
      {"path": "tests/payment/test_concurrency.py"}]),
    ("Add export CSV feature", "Download filtered results as CSV.",
     [{"path": "src/api/export.py"}, {"path": "src/services/csv_writer.py"},
      {"path": "frontend/src/components/ExportButton.tsx"}, {"path": "src/api/admin.py"},
      {"path": "Dockerfile"}]),
    ("Webhook retry logic broken", "Retries once instead of exponential backoff.",
     [{"path": "src/webhooks/delivery.py"}, {"path": "src/webhooks/retry.py"},
      {"path": "src/config/webhook_settings.py"}, {"path": "tests/webhooks/test_retry.py"},
      {"path": "src/utils/backoff.py"}]),
    ("Add rate limiting auth", "No rate limit on login endpoints.",
     [{"path": "src/middleware/rate_limiter.py"}, {"path": "src/api/auth.py"},
      {"path": "src/config/rate_limits.yaml"}, {"path": "src/app.py"},
      {"path": "tests/test_rate_limit.py"}]),
    # Repeat 10 more with varied files to reach 20
    ("Timeout on image upload", "Images > 20MB timeout.",
     [{"path": "src/api/upload.py"}, {"path": "src/services/image_processor.py"},
      {"path": "config/nginx.conf"}, {"path": "src/utils/timeout.py"},
      {"path": "Dockerfile"}]),
    ("Fix typos in error messages", "Spelling mistakes in error strings.",
     [{"path": "src/utils/errors.py"}, {"path": "src/api/messages.py"},
      {"path": "frontend/src/i18n/en.json"}, {"path": "README.md"},
      {"path": "tests/test_errors.py"}]),
    ("Upgrade Node 18 to 22", "Node 18 EOL, need to upgrade.",
     [{"path": ".nvmrc"}, {"path": "Dockerfile"}, {"path": ".github/workflows/ci.yml"},
      {"path": "package.json"}, {"path": "frontend/tsconfig.json"}]),
    ("Add health check endpoint", "Need /health for K8s probe.",
     [{"path": "src/api/health.py"}, {"path": "src/app.py"},
      {"path": "k8s/deployment.yaml"}, {"path": "tests/test_health.py"},
      {"path": "src/db/connection.py"}]),
    ("Session cookie HttpOnly", "Auth cookie missing HttpOnly flag.",
     [{"path": "src/middleware/session.py"}, {"path": "src/config/auth.py"},
      {"path": "src/api/login.py"}, {"path": "tests/test_auth.py"},
      {"path": "src/utils/cookies.py"}]),
    ("Dependency vulnerability lodash", "CVE-2024 in lodash 4.17.20.",
     [{"path": "package.json"}, {"path": "package-lock.json"},
      {"path": "frontend/src/utils/debounce.ts"}, {"path": "yarn.lock"},
      {"path": ".github/dependabot.yml"}]),
    ("Localization Chinese translations", "Need i18n with Chinese.",
     [{"path": "frontend/src/i18n/index.ts"}, {"path": "frontend/src/i18n/zh-CN.json"},
      {"path": "frontend/src/i18n/en.json"}, {"path": "frontend/src/components/*.tsx"},
      {"path": "frontend/package.json"}]),
    ("Feature flag cleanup", "Remove always-on experiment flags.",
     [{"path": "src/config/features.py"}, {"path": "src/components/OldExperiment.tsx"},
      {"path": "src/components/NewExperiment.tsx"}, {"path": "src/api/experiment.py"},
      {"path": "tests/test_features.py"}]),
    ("API 500 on empty body", "Empty request body crashes server.",
     [{"path": "src/api/submit.py"}, {"path": "src/middleware/validation.py"},
      {"path": "src/utils/request_parser.py"}, {"path": "tests/api/test_submit.py"},
      {"path": "src/app.py"}]),
    ("Add export to PDF", "Users want PDF download of reports.",
     [{"path": "src/services/pdf_generator.py"}, {"path": "src/api/reports.py"},
      {"path": "src/templates/report.html"}, {"path": "requirements.txt"},
      {"path": "tests/reports/test_pdf.py"}]),
]


async def test_classify():
    """Test classify_issue 20 times."""
    results = {"first_try": 0, "retry_success": 0, "fallback": 0, "errors": []}
    model = _config()[2]
    print(f"Classification test — model={model}, n={len(CLASSIFY_INPUTS)}")

    for i, (title, body) in enumerate(CLASSIFY_INPUTS):
        system = (
            "You are a software engineering triage assistant. "
            "ONLY return valid JSON, no markdown, no explanation outside the JSON. "
            "Keys: type (bug|feature|docs|test|security), "
            "severity (low|medium|high), confidence (0.0-1.0), reasoning (string)."
        )
        user = f"Issue title: {title}\n\nIssue body:\n{body}"

        raw = await llm_call(system, user)
        try:
            Classification.model_validate(raw)
            results["first_try"] += 1
            print(f"  [{i+1:2d}] ✓ first try")
        except ValidationError as e:
            # Retry
            retry_user = (
                f"Your response did not match the required schema. Errors: {e.errors()}\n\n"
                f"Original request:\n{user}\n\n"
                "Return ONLY valid JSON matching the required keys and types."
            )
            raw2 = await llm_call(system, retry_user)
            try:
                Classification.model_validate(raw2)
                results["retry_success"] += 1
                print(f"  [{i+1:2d}] ✓ retry success — errors: {e.errors()[:2]}")
            except ValidationError as e2:
                results["fallback"] += 1
                print(f"  [{i+1:2d}] ✗ FALLBACK — 1st: {e.errors()[:2]}, 2nd: {e2.errors()[:2]}")

    return results


async def test_rank_files():
    """Test rank_files 20 times."""
    results = {"first_try": 0, "retry_success": 0, "fallback": 0, "errors": []}
    model = _config()[2]
    print(f"\nFileRanking test — model={model}, n={len(RANK_FILES)}")

    for i, (title, body, files) in enumerate(RANK_FILES):
        file_list = "\n".join(f"- {f['path']}" for f in files)
        system = (
            "You are a code reviewer. Given a GitHub issue and a list of file paths, "
            "ONLY return valid JSON, no markdown. Output JSON with key 'files': "
            "an array of objects, each with: "
            "path (string), relevance_score (0.0-1.0), reason (string). "
            "Order by relevance_score descending."
        )
        user = f"Issue: {title}\n\nDescription:\n{body}\n\nFiles:\n{file_list}"

        raw = await llm_call(system, user)
        try:
            FileRanking.model_validate(raw)
            results["first_try"] += 1
            print(f"  [{i+1:2d}] ✓ first try")
        except ValidationError as e:
            retry_user = (
                f"Your response did not match the required schema. Errors: {e.errors()}\n\n"
                f"Original request:\n{user}\n\n"
                "Return ONLY valid JSON matching the required keys and types."
            )
            raw2 = await llm_call(system, retry_user)
            try:
                FileRanking.model_validate(raw2)
                results["retry_success"] += 1
                print(f"  [{i+1:2d}] ✓ retry success — errors: {e.errors()[:2]}")
            except ValidationError as e2:
                results["fallback"] += 1
                print(f"  [{i+1:2d}] ✗ FALLBACK — 1st: {e.errors()[:2]}, 2nd: {e2.errors()[:2]}")

    return results


async def test_fix_plan():
    """Test generate_fix_plan 20 times."""
    results = {"first_try": 0, "retry_success": 0, "fallback": 0, "errors": []}
    model = _config()[2]
    print(f"\nFixPlan test — model={model}, n={len(CLASSIFY_INPUTS)}")

    for i, (title, body) in enumerate(CLASSIFY_INPUTS[:20]):
        classification = {"type": "bug", "severity": "medium", "confidence": 0.85, "reasoning": "Test input"}
        files_summary = "- src/api/handler.py (relevance: 0.9): Main request handler\n- src/db/models.py (relevance: 0.7): Database models"
        system = (
            "You are a senior software engineer. Given a GitHub issue analysis, "
            "return JSON with keys: fix_plan (string, markdown), "
            "risk_level (low|medium|high), test_suggestions (array of strings)."
        )
        user = (
            f"Issue: {title}\n\n"
            f"Description:\n{body}\n\n"
            f"Classification: {classification}\n\n"
            f"Relevant files:\n{files_summary}"
        )

        raw = await llm_call(system, user)
        try:
            validated = FixPlan.model_validate(raw)
            # Also check that test_suggestions is a list of strings
            results["first_try"] += 1
            print(f"  [{i+1:2d}] ✓ first try")
        except ValidationError as e:
            retry_user = (
                f"Your response did not match the required schema. Errors: {e.errors()}\n\n"
                f"Original request:\n{user}\n\n"
                "Return ONLY valid JSON matching the required keys and types."
            )
            raw2 = await llm_call(system, retry_user)
            try:
                FixPlan.model_validate(raw2)
                results["retry_success"] += 1
                print(f"  [{i+1:2d}] ✓ retry success — errors: {e.errors()[:2]}")
            except ValidationError as e2:
                results["fallback"] += 1
                print(f"  [{i+1:2d}] ✗ FALLBACK — 1st: {e.errors()[:2]}, 2nd: {e2.errors()[:2]}")

    return results


def print_summary(name, r, total):
    pct1 = r["first_try"] / total * 100
    pct2 = r["retry_success"] / total * 100
    pct3 = r["fallback"] / total * 100
    print(f"\n{'='*60}")
    print(f"  {name}: {r['first_try']}/{total} first-try ({pct1:.0f}%)")
    print(f"           {r['retry_success']}/{total} retry success ({pct2:.0f}%)")
    print(f"           {r['fallback']}/{total} fallback ({pct3:.0f}%)")
    print(f"  Overall pass (first+retry): {r['first_try']+r['retry_success']}/{total} ({(r['first_try']+r['retry_success'])/total*100:.0f}%)")


async def main():
    print("Pydantic Validation Benchmark — deepseek-v4-pro")
    print("=" * 60)
    start = time.time()

    r1 = await test_classify()
    r2 = await test_rank_files()
    r3 = await test_fix_plan()

    elapsed = time.time() - start

    print_summary("Classification", r1, len(CLASSIFY_INPUTS))
    print_summary("FileRanking  ", r2, len(RANK_FILES))
    print_summary("FixPlan      ", r3, min(20, len(CLASSIFY_INPUTS)))

    total_calls = len(CLASSIFY_INPUTS) + len(RANK_FILES) + 20
    first_try_total = r1["first_try"] + r2["first_try"] + r3["first_try"]
    fallback_total = r1["fallback"] + r2["fallback"] + r3["fallback"]
    retry_total = r1["retry_success"] + r2["retry_success"] + r3["retry_success"]

    print(f"\n{'='*60}")
    print(f"  OVERALL: {first_try_total}/{total_calls} first-try ({first_try_total/total_calls*100:.0f}%)")
    print(f"           {retry_total}/{total_calls} retry success ({retry_total/total_calls*100:.0f}%)")
    print(f"           {fallback_total}/{total_calls} fallback ({fallback_total/total_calls*100:.0f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/total_calls:.1f}s/call avg)")
    print(f"  Model: {_config()[2]}")


if __name__ == "__main__":
    asyncio.run(main())
