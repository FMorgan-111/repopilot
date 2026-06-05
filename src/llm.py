import json
import os
import re
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response — handles markdown, code fences, raw text."""
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try ```json ... ``` block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try first { ... } block
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def _config() -> tuple[str, str, str]:
    """Return (api_key, base_url, model) from environment. base_url includes /v1."""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    return api_key, base_url, model


async def llm_call(system_prompt: str, user_prompt: str, model: str = None) -> dict:
    """Call an OpenAI-compatible chat endpoint and return parsed JSON."""
    api_key, base_url, default_model = _config()
    if model is None:
        model = default_model
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_json(content)


async def classify_issue(title: str, body: str) -> dict:
    """Classify issue type, severity, and confidence."""
    system = (
        "You are a software engineering triage assistant. "
        "ONLY return valid JSON, no markdown, no explanation outside the JSON. "
        "Keys: type (bug|feature|docs|test|security), "
        "severity (low|medium|high), confidence (0.0-1.0), reasoning (string)."
    )
    user = f"Issue title: {title}\n\nIssue body:\n{body}"
    return await llm_call(system, user)


async def rank_files(issue_title: str, issue_body: str, files: list[dict]) -> list[dict]:
    """Rank files by relevance to the issue, return list with relevance_score and reason."""
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
    result = await llm_call(system, user)
    return result.get("files", [])


async def generate_fix_plan(
    issue_title: str,
    issue_body: str,
    classification: dict,
    ranked_files: list[dict],
) -> dict:
    """Generate a fix plan, risk level, and test suggestions."""
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
    return await llm_call(system, user)
