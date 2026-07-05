import json
import os
import re
import warnings

from pydantic import BaseModel, ValidationError

from .http_client import llm_request
from .schemas import Classification, FileRanking, FixPlan


def _repair_json_text(text: str) -> str:
    """Best-effort fixes for JSON mistakes chat models commonly make, applied
    only as a fallback after a strict parse fails: smart quotes used as
    delimiters and trailing commas before a closing bracket. Deliberately does
    NOT strip // comments — that would corrupt URLs inside string values."""
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # trailing commas
    return text


def _try_loads(candidate: str) -> dict | None:
    """Parse a candidate strictly, then with a lenient repair pass."""
    for text in (candidate, _repair_json_text(candidate)):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response — handles markdown, code fences, raw text."""
    # Try raw parse first.
    parsed = _try_loads(text)
    if parsed is not None:
        return parsed
    # Try ```json ... ``` block.
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        parsed = _try_loads(m.group(1))
        if parsed is not None:
            return parsed
    # Try first { ... } block with bracket counting (handles arbitrary nesting).
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    parsed = _try_loads(text[start:i + 1])
                    if parsed is not None:
                        return parsed
                    break
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


async def llm_call(system_prompt: str, user_prompt: str, model: str = None) -> dict:
    """Call an OpenAI-compatible chat endpoint and return parsed JSON.

    On an unparseable first response, retry once with an explicit instruction to
    emit only JSON — some models bury the JSON in prose on the first try."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp_data = await llm_request(messages, model)
    content = resp_data["choices"][0]["message"]["content"]
    try:
        return _extract_json(content)
    except ValueError:
        retry_messages = messages + [
            {"role": "assistant", "content": content[:2000]},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Reply with ONLY "
                    "a single valid JSON object — no prose, no code fences."
                ),
            },
        ]
        resp_data = await llm_request(retry_messages, model)
        content = resp_data["choices"][0]["message"]["content"]
        return _extract_json(content)


async def classify_issue(title: str, body: str) -> dict:
    """Classify issue type, severity, and confidence (Pydantic-validated)."""
    system = (
        "You are a software engineering triage assistant. "
        "ONLY return valid JSON, no markdown, no explanation outside the JSON. "
        "Keys: type (bug|feature|docs|test|security), "
        "severity (low|medium|high), confidence (0.0-1.0), reasoning (string)."
    )
    user = f"Issue title: {title}\n\nIssue body:\n{body}"
    return await validate_or_retry(system, user, Classification)


async def rank_files(issue_title: str, issue_body: str, files: list[dict]) -> list[dict]:
    """Rank files by relevance (Pydantic-validated)."""
    if not files:
        return []
    file_list = "\n".join(f"- {f['path']}" for f in files)
    system = (
        "You are a code reviewer. Given a GitHub issue and a list of file paths, "
        "ONLY return valid JSON, no markdown. Output JSON with key 'files': "
        "an array of objects, each with: "
        "path (string), relevance_score (0.0-1.0), reason (string). "
        "Order by relevance_score descending."
    )
    user = (
        f"Issue: {issue_title}\n\nDescription:\n{issue_body}\n\nFiles:\n{file_list}"
    )
    result = await validate_or_retry(system, user, FileRanking)
    return result.get("files", [])


async def generate_fix_plan(
    issue_title: str,
    issue_body: str,
    classification: dict,
    ranked_files: list[dict],
) -> dict:
    """Generate a fix plan (Pydantic-validated)."""
    files_summary = "\n".join(
        f"- {f.get('path', '?')} (relevance: {f.get('relevance_score', '?')}): {f.get('reason', '')}"
        for f in ranked_files[:5]
    )
    system = (
        "You are a senior software engineer. Given a GitHub issue analysis, "
        "return JSON with keys: fix_plan (string, markdown), "
        "risk_level (low|medium|high), test_suggestions (array of strings)."
    )
    user = (
        f"Issue: {issue_title}\n\n"
        f"Description:\n{issue_body}\n\n"
        f"Classification: {json.dumps(classification)}\n\n"
        f"Relevant files:\n{files_summary}"
    )
    return await validate_or_retry(system, user, FixPlan)


async def validate_or_retry(system: str, user: str, schema: type[BaseModel]) -> dict:
    """Call LLM, validate with Pydantic, retry once on failure."""
    raw = await llm_call(system, user)
    try:
        validated = schema.model_validate(raw)
        return validated.model_dump()
    except ValidationError as e:
        # Retry once with schema-error feedback (JSON parsed fine; schema didn't match)
        errors = str(e.errors())
        retry_user = (
            f"Your response did not match the required schema. Errors: {errors}\n\n"
            f"Original request:\n{user}\n\n"
            "Return ONLY valid JSON matching the required keys and types."
        )
        raw2 = await llm_call(system, retry_user)
        try:
            validated2 = schema.model_validate(raw2)
            return validated2.model_dump()
        except ValidationError as e2:
            warnings.warn(f"validate_or_retry: schema {schema.__name__} failed after retry: {e2.errors()}")
            return raw2  # Fallback: return last raw dict (closer to correct shape than first)
