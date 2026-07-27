"""RepoPilot — AI Agent that turns GitHub issues into fix plans."""
# ruff: noqa: E402,I001
import hmac
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
load_dotenv(override=False)

from fastapi import FastAPI, HTTPException, Path
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from src.agent import analyze_issue
from src.agent_loop import agent_analyze
from src.async_safety import CancellationDrainError
from src.new_agent import agent_v2, intelligent_analyze_issue, resume_agent_v2
from src.run_store import (
    ResumeConflictError,
    format_replay_markdown,
    load_run,
    summarize_replay,
    summarize_run,
)
from src.state import (
    DEFAULT_AGENT_V2_MAX_RETRIES,
    DEFAULT_AGENT_V2_TOKEN_BUDGET,
    MAX_AGENT_V2_MAX_RETRIES,
    AgentState,
)

_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_ISSUE_PATH_RE = re.compile(
    r"^/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)/"
    r"([A-Za-z0-9_.-]{1,100})/issues/([1-9][0-9]*)$"
)
_RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
MAX_REQUEST_BODY_BYTES = 65_536
MAX_PROTECTED_REQUESTS = 2


class _ProtectedRequestSlots:
    """A loop-independent, non-queueing process-local concurrency gate."""

    def __init__(self, limit: int):
        self._limit = limit
        self._active = 0
        self._lock = threading.Lock()

    @asynccontextmanager
    async def claim(self):
        with self._lock:
            if self._active >= self._limit:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests.",
                    headers={"Retry-After": "1"},
                )
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1


_protected_request_slots = _ProtectedRequestSlots(MAX_PROTECTED_REQUESTS)


class ApiBoundaryMiddleware:
    """Authenticate every non-health HTTP request before reading its body."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method") == "GET" and scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return

        configured_token = os.getenv("REPOPILOT_API_TOKEN")
        if configured_token is None or not configured_token.strip():
            await _error_response(503, "Service unavailable.")(
                scope, receive, send
            )
            return

        authorization_headers = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"authorization"
        ]
        provided_token = None
        if len(authorization_headers) == 1:
            try:
                authorization = authorization_headers[0].decode("ascii")
            except UnicodeDecodeError:
                authorization = ""
            match = re.fullmatch(r"Bearer ([^\s]+)", authorization)
            if match:
                provided_token = match.group(1)

        if provided_token is None or not hmac.compare_digest(
            provided_token.encode("ascii"), configured_token.encode("utf-8")
        ):
            await _error_response(
                401,
                "Unauthorized.",
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
            return

        content_lengths = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        if len(content_lengths) > 1:
            await _error_response(400, "Invalid request.")(scope, receive, send)
            return
        if content_lengths:
            try:
                declared = content_lengths[0].decode("ascii")
            except UnicodeDecodeError:
                declared = ""
            if not re.fullmatch(r"[0-9]+", declared):
                await _error_response(400, "Invalid request.")(
                    scope, receive, send
                )
                return
            normalized_length = declared.lstrip("0") or "0"
            maximum_length = str(MAX_REQUEST_BODY_BYTES)
            if (
                len(normalized_length) > len(maximum_length)
                or (
                    len(normalized_length) == len(maximum_length)
                    and normalized_length > maximum_length
                )
            ):
                await _error_response(413, "Request body too large.")(
                    scope, receive, send
                )
                return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                await _error_response(400, "Invalid request.")(
                    scope, receive, send
                )
                return
            chunk = message.get("body", b"")
            if len(chunk) > MAX_REQUEST_BODY_BYTES - len(body):
                await _error_response(413, "Request body too large.")(
                    scope, receive, send
                )
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        body_sent = False

        async def bounded_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, bounded_receive, send)


def _error_response(
    status_code: int,
    detail: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"detail": detail},
        status_code=status_code,
        headers=headers,
    )


@dataclass(frozen=True)
class RepositoryIdentity:
    owner: str
    repo: str

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.repo}".casefold()


def _parse_repository_entry(value: str) -> RepositoryIdentity:
    if value != value.strip() or value.count("/") != 1:
        raise ValueError("invalid repository entry")
    owner, repo = value.split("/", 1)
    if (
        not _OWNER_RE.fullmatch(owner)
        or not _REPOSITORY_RE.fullmatch(repo)
        or owner in {".", ".."}
        or repo in {".", ".."}
    ):
        raise ValueError("invalid repository entry")
    return RepositoryIdentity(owner=owner, repo=repo)


def _allowed_repositories() -> frozenset[str]:
    raw = os.getenv("REPOPILOT_ALLOWED_REPOS")
    if raw is None or not raw:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    try:
        entries = [_parse_repository_entry(item) for item in raw.split(",")]
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail="Service unavailable."
        ) from exc
    keys = [entry.key for entry in entries]
    if not keys or len(keys) != len(set(keys)):
        raise HTTPException(status_code=503, detail="Service unavailable.")
    return frozenset(keys)


def _parse_issue_repository(issue_url: str) -> RepositoryIdentity:
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in issue_url):
        raise HTTPException(status_code=400, detail="Invalid request.")
    try:
        parsed = urlsplit(issue_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid request.") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(status_code=400, detail="Invalid request.")
    match = _ISSUE_PATH_RE.fullmatch(parsed.path)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid request.")
    owner, repo, _issue_number = match.groups()
    if owner in {".", ".."} or repo in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid request.")
    return RepositoryIdentity(owner=owner, repo=repo)


def _authorize_issue_url(issue_url: str) -> RepositoryIdentity:
    allowed = _allowed_repositories()
    repository = _parse_issue_repository(issue_url)
    if repository.key not in allowed:
        raise HTTPException(
            status_code=403,
            detail="Repository is not authorized.",
        )
    return repository


def _load_authorized_run(run_id: str) -> AgentState:
    allowed = _allowed_repositories()
    try:
        state = load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Saved run was not found.",
        ) from exc
    except (json.JSONDecodeError, ValidationError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Saved run could not be loaded.",
        ) from exc

    if (
        state.trace_id != run_id
        or not (0 <= state.max_retries <= MAX_AGENT_V2_MAX_RETRIES)
        or not (1 <= state.token_budget <= 100_000)
    ):
        raise HTTPException(
            status_code=500,
            detail="Saved run could not be loaded.",
        )

    try:
        repository = _parse_issue_repository(state.issue_url)
    except HTTPException as exc:
        raise HTTPException(
            status_code=403,
            detail="Repository is not authorized.",
        ) from exc

    if (
        (state.owner and state.owner.casefold() != repository.owner.casefold())
        or (state.repo and state.repo.casefold() != repository.repo.casefold())
        or repository.key not in allowed
    ):
        raise HTTPException(
            status_code=403,
            detail="Repository is not authorized.",
        )
    return state


app = FastAPI(
    title="RepoPilot",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(ApiBoundaryMiddleware)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_request, _exc):
    return _error_response(422, "Invalid request.")


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeRequest(StrictRequestModel):
    issue_url: str = Field(max_length=2_048)


class AgentRequest(StrictRequestModel):
    issue_url: str = Field(max_length=2_048)
    max_turns: int = Field(default=10, ge=1, le=10, strict=True)


class IntelligentAgentRequest(StrictRequestModel):
    issue_url: str = Field(max_length=2_048)
    max_turns: int = Field(default=10, ge=1, le=10, strict=True)
    token_budget: int = Field(
        default=DEFAULT_AGENT_V2_TOKEN_BUDGET,
        ge=1,
        le=100_000,
        strict=True,
    )


class AgentV2Request(StrictRequestModel):
    issue_url: str = Field(max_length=2_048)
    max_retries: int = Field(
        default=DEFAULT_AGENT_V2_MAX_RETRIES,
        ge=0,
        le=MAX_AGENT_V2_MAX_RETRIES,
        strict=True,
    )
    token_budget: int = Field(
        default=DEFAULT_AGENT_V2_TOKEN_BUDGET,
        ge=1,
        le=100_000,
        strict=True,
    )


class AgentV2ResumeRequest(StrictRequestModel):
    run_id: str = Field(pattern=_RUN_ID_PATTERN, max_length=64)
    human_answer: str = Field(min_length=1, max_length=16_384)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """基础线性 pipeline 分析"""
    _authorize_issue_url(req.issue_url)
    async with _protected_request_slots.claim():
        try:
            result = await analyze_issue(req.issue_url)
        except Exception:
            return _error_response(502, "Agent request failed.")
    if "error" in result:
        status = 400 if "Invalid" in result["error"] else 502
        return _error_response(status, "Agent request failed.")
    return result


@app.post("/agent")
async def agent(req: AgentRequest):
    """简单 LLM 循环 agent"""
    _authorize_issue_url(req.issue_url)
    async with _protected_request_slots.claim():
        try:
            result = await agent_analyze(req.issue_url, req.max_turns)
        except Exception:
            return _error_response(502, "Agent request failed.")
    if "error" in result:
        status = 400 if "Invalid" in result["error"] else 502
        return _error_response(status, "Agent request failed.")
    return result


@app.post("/intelligent-agent")
async def intelligent_agent(req: IntelligentAgentRequest):
    """🚀 新的智能推理 agent - 带状态机和执行反馈循环"""
    _authorize_issue_url(req.issue_url)
    async with _protected_request_slots.claim():
        try:
            result = await intelligent_analyze_issue(
                req.issue_url,
                max_retries=min(req.max_turns, MAX_AGENT_V2_MAX_RETRIES),
                token_budget=req.token_budget,
            )
        except CancellationDrainError:
            raise
        except Exception:
            return _error_response(502, "Agent request failed.")
    if result.get("error"):
        status = 400 if "Invalid" in result["error"] else 502
        return _error_response(status, "Agent request failed.")
    return result


@app.post("/agent/v2")
async def agent_v2_endpoint(req: AgentV2Request):
    """State-graph agent with execute/test/replan feedback loop."""
    _authorize_issue_url(req.issue_url)
    async with _protected_request_slots.claim():
        try:
            result = await agent_v2(
                req.issue_url,
                max_retries=req.max_retries,
                token_budget=req.token_budget,
            )
        except CancellationDrainError:
            raise
        except Exception:
            return _error_response(502, "Agent request failed.")
    if result.get("error"):
        status = 400 if "Invalid" in result["error"] else 502
        return _error_response(status, "Agent request failed.")
    return result


@app.post("/agent/v2/resume")
async def agent_v2_resume_endpoint(req: AgentV2ResumeRequest):
    """Resume a paused state-graph agent run with human input."""
    state = _load_authorized_run(req.run_id)
    async with _protected_request_slots.claim():
        try:
            result = await resume_agent_v2(
                req.run_id,
                req.human_answer,
                state=state,
            )
        except ResumeConflictError:
            return _error_response(409, "Run resume conflict.")
        except CancellationDrainError:
            raise
        except Exception:
            return _error_response(502, "Agent request failed.")

    if result.get("error"):
        status = 400 if _is_client_error(result["error"]) else 502
        return _error_response(status, "Agent request failed.")
    return result


@app.get("/agent/v2/runs/{run_id}")
async def agent_v2_inspect_endpoint(
    run_id: str = Path(pattern=_RUN_ID_PATTERN, max_length=64),
):
    """Inspect the bounded summary of an authorized saved run."""
    state = _load_authorized_run(run_id)
    async with _protected_request_slots.claim():
        try:
            return summarize_run(state)
        except Exception:
            return _error_response(500, "Saved run could not be loaded.")


@app.get("/agent/v2/runs/{run_id}/replay")
async def agent_v2_replay_endpoint(
    run_id: str = Path(pattern=_RUN_ID_PATTERN, max_length=64),
    format: Literal["json", "markdown"] = "json",
):
    """Replay the white-box decision trace for a saved run."""
    state = _load_authorized_run(run_id)
    async with _protected_request_slots.claim():
        try:
            replay = summarize_replay(state)
            if format == "markdown":
                return PlainTextResponse(
                    format_replay_markdown(replay),
                    media_type="text/markdown",
                )
            return replay
        except Exception:
            return _error_response(500, "Saved run could not be loaded.")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _is_client_error(error: str) -> bool:
    return "Invalid" in error or "not waiting for user input" in error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
