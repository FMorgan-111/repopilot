"""Quick benchmark: 3 samples per schema type."""
import asyncio, json, sys
sys.path.insert(0, ".")
from src.llm import llm_call
from src.schemas import Classification, FileRanking, FixPlan
from pydantic import ValidationError

CASES = [
    ("Classification", Classification,
     "ONLY return JSON. Keys: type(bug|feature|docs|test|security), severity(low|medium|high), confidence(0.0-1.0), reasoning(string).",
     "Issue: Login button broken on mobile. Users cannot log in."),
    ("Classification", Classification,
     "ONLY return JSON. Keys: type(bug|feature|docs|test|security), severity(low|medium|high), confidence(0.0-1.0), reasoning(string).",
     "Issue: Add dark mode toggle to settings panel."),
    ("Classification", Classification,
     "ONLY return JSON. Keys: type(bug|feature|docs|test|security), severity(low|medium|high), confidence(0.0-1.0), reasoning(string).",
     "Issue: SQL injection in search — unparameterized user input."),
    ("FileRanking", FileRanking,
     "ONLY return JSON. Keys: files(array of {path(string),relevance_score(0.0-1.0),reason(string)}).",
     "Issue: Memory leak in websocket.\nFiles:\n- src/ws/handler.py\n- src/ws/connection.py\n- tests/ws_test.py"),
    ("FileRanking", FileRanking,
     "ONLY return JSON. Keys: files(array of {path(string),relevance_score(0.0-1.0),reason(string)}).",
     "Issue: Add CSV export button to dashboard.\nFiles:\n- src/api/export.py\n- frontend/ExportButton.tsx\n- README.md"),
    ("FileRanking", FileRanking,
     "ONLY return JSON. Keys: files(array of {path(string),relevance_score(0.0-1.0),reason(string)}).",
     "Issue: Pagination broken, page 2 returns empty.\nFiles:\n- src/api/pagination.py\n- src/db/queries.py\n- frontend/Paginator.tsx"),
    ("FixPlan", FixPlan,
     "ONLY return JSON. Keys: fix_plan(string,markdown), risk_level(low|medium|high), test_suggestions(array of strings).",
     "Issue: Race condition in payment — double charge on concurrent clicks.\nClassification: {\"type\":\"bug\",\"severity\":\"high\",\"confidence\":0.9}\nFiles:\n- src/payment/processor.py\n- src/payment/idempotency.py"),
    ("FixPlan", FixPlan,
     "ONLY return JSON. Keys: fix_plan(string,markdown), risk_level(low|medium|high), test_suggestions(array of strings).",
     "Issue: Add rate limiting to auth endpoints.\nClassification: {\"type\":\"security\",\"severity\":\"medium\",\"confidence\":0.85}\nFiles:\n- src/middleware/rate_limit.py\n- src/api/auth.py"),
    ("FixPlan", FixPlan,
     "ONLY return JSON. Keys: fix_plan(string,markdown), risk_level(low|medium|high), test_suggestions(array of strings).",
     "Issue: Upgrade Node.js 18 to 22.\nClassification: {\"type\":\"feature\",\"severity\":\"low\",\"confidence\":0.95}\nFiles:\n- .nvmrc\n- Dockerfile\n- .github/workflows/ci.yml"),
]


async def main():
    stats = {}
    for schema_name, schema, system, user in CASES:
        if schema_name not in stats:
            stats[schema_name] = {"first": 0, "retry": 0, "fallback": 0, "n": 0}
        stats[schema_name]["n"] += 1
        n = stats[schema_name]["n"]

        raw = await llm_call(system, user + "\n\nReturn ONLY JSON, no markdown, no explanation.")
        try:
            schema.model_validate(raw)
            stats[schema_name]["first"] += 1
            print(f"  [{schema_name} #{n}] PASS first try", flush=True)
        except ValidationError as e:
            errs = str(e.errors())[:250]
            retry = await llm_call(
                system,
                f"Validation errors: {errs}\n\nOriginal request:\n{user}\n\nReturn ONLY valid JSON matching the schema. No markdown."
            )
            try:
                schema.model_validate(retry)
                stats[schema_name]["retry"] += 1
                print(f"  [{schema_name} #{n}] PASS retry — err: {errs[:120]}", flush=True)
            except ValidationError as e2:
                stats[schema_name]["fallback"] += 1
                print(f"  [{schema_name} #{n}] FALLBACK — 1st: {errs[:100]}, 2nd: {str(e2.errors())[:100]}", flush=True)

    print("\n" + "=" * 55)
    for name, s in stats.items():
        total = s["n"]
        p1 = s["first"] / total * 100
        p2 = s["retry"] / total * 100
        p3 = s["fallback"] / total * 100
        print(f"  {name}: {s['first']}/{total} first ({p1:.0f}%), {s['retry']}/{total} retry ({p2:.0f}%), {s['fallback']}/{total} fallback ({p3:.0f}%)")
    total_all = sum(s["n"] for s in stats.values())
    first_all = sum(s["first"] for s in stats.values())
    fallback_all = sum(s["fallback"] for s in stats.values())
    retry_all = sum(s["retry"] for s in stats.values())
    print(f"  TOTAL: {first_all}/{total_all} first ({first_all/total_all*100:.0f}%), {retry_all}/{total_all} retry ({retry_all/total_all*100:.0f}%), {fallback_all}/{total_all} fallback ({fallback_all/total_all*100:.0f}%)")


asyncio.run(main())
