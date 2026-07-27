"""Independent configuration for primary and escalation model providers."""

from __future__ import annotations

import os
import re
import warnings
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, SecretStr

ProviderName = Literal["primary", "escalation"]

PRIMARY_BASE_URL = "https://linoapi.com.cn/v1"
PRIMARY_MODEL = "gemini-3.5-flash:stable"
ESCALATION_BASE_URL = "https://linoapi.com.cn/v1"
ESCALATION_MODEL = "claude-opus-4-8:stable"


class ModelConfig(BaseModel):
    """Credentials and endpoint settings for one OpenAI-compatible provider."""

    provider: ProviderName
    model: str
    base_url: str
    api_key: SecretStr


def _validated_base_url(value: str, variable: str) -> str:
    candidate = value.strip().rstrip("/")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{variable} contains an invalid base URL") from exc
    if (
        not candidate
        or any(character.isspace() for character in candidate)
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError(f"{variable} contains an unsafe base URL")
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError(f"{variable} base URL must use HTTPS")
    return candidate


def _primary_base_url() -> str:
    generic = os.getenv("LLM_BASE_URL", "").strip()
    legacy = os.getenv("OPENAI_BASE_URL", "").strip()
    generic_url = _validated_base_url(generic, "LLM_BASE_URL") if generic else ""
    legacy_url = _validated_base_url(legacy, "OPENAI_BASE_URL") if legacy else ""
    if generic_url and legacy_url and generic_url != legacy_url:
        raise ValueError("conflicting LLM_BASE_URL and OPENAI_BASE_URL values")
    if generic_url:
        return generic_url
    if legacy_url:
        warnings.warn(
            "OPENAI_BASE_URL is deprecated; use LLM_BASE_URL",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy_url
    return _validated_base_url(PRIMARY_BASE_URL, "LLM_BASE_URL")


def get_model_name(provider: ProviderName) -> str:
    """Resolve non-secret model metadata without constructing request config."""
    if provider == "primary":
        return os.getenv("LLM_MODEL", PRIMARY_MODEL)
    return os.getenv("LLM_ESCALATION_MODEL", ESCALATION_MODEL)


def get_model_config(provider: ProviderName) -> ModelConfig:
    """Resolve one provider's configuration without crossing credential domains."""
    if provider == "primary":
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is required for the primary provider")
        return ModelConfig(
            provider="primary",
            model=get_model_name("primary"),
            base_url=_primary_base_url(),
            api_key=SecretStr(api_key),
        )
    return ModelConfig(
        provider="escalation",
        model=get_model_name("escalation"),
        base_url=_validated_base_url(
            os.getenv("LLM_ESCALATION_BASE_URL", ESCALATION_BASE_URL),
            "LLM_ESCALATION_BASE_URL",
        ),
        api_key=SecretStr(os.getenv("LLM_ESCALATION_API_KEY", "").strip()),
    )


def escalation_is_configured() -> bool:
    """Return whether explicitly enabled escalation has its own API key."""
    return (
        os.getenv("REPOPILOT_ESCALATION_ENABLED") == "1"
        and bool(get_model_config("escalation").api_key.get_secret_value())
    )


_BEARER_SECRET = re.compile(r"(?i)(\bBearer\s+)[^\s,;]+")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[-_]?key|access[-_]?token|token|key|authorization)=)[^&#\s;]+"
)
_HEADER_SECRET = re.compile(
    r"(?i)(\b(?:authorization|proxy-authorization|x-api-key|api[-_]?key|token)\s*[:=]\s*)(?:Bearer\s+)?[^\s,;]+"
)
_QUOTED_FIELD_SECRET = re.compile(
    r"(?i)((?:[\"'](?:authorization|proxy-authorization|x-api-key|api[-_]?key|access[-_]?token|token)[\"'])\s*[:=]\s*[\"'])(?:Bearer\s+)?[^\"']*[\"']"
)
_BARE_SK_SECRET = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])")


def redact_secrets(text: str) -> str:
    """Replace credential-shaped values before text leaves a local boundary."""
    message = str(text)
    message = _BEARER_SECRET.sub(r"\1[REDACTED]", message)
    message = _QUERY_SECRET.sub(r"\1[REDACTED]", message)
    message = _QUOTED_FIELD_SECRET.sub(r"\1[REDACTED]", message)
    message = _HEADER_SECRET.sub(r"\1[REDACTED]", message)
    return _BARE_SK_SECRET.sub("[REDACTED]", message)


def sanitize_provider_error(exc: BaseException) -> str:
    """Return an exception message with common credentials replaced safely."""
    return redact_secrets(str(exc))
