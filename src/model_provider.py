"""Independent configuration for primary and escalation model providers."""

from __future__ import annotations

import os
import re
from typing import Literal

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


def get_model_config(provider: ProviderName) -> ModelConfig:
    """Resolve one provider's configuration without crossing credential domains."""
    if provider == "primary":
        return ModelConfig(
            provider="primary",
            model=os.getenv("LLM_MODEL", PRIMARY_MODEL),
            base_url=os.getenv("OPENAI_BASE_URL", PRIMARY_BASE_URL).rstrip("/"),
            api_key=SecretStr(
                os.getenv("LINOAPI_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY", "")
            ),
        )
    return ModelConfig(
        provider="escalation",
        model=os.getenv("LLM_ESCALATION_MODEL", ESCALATION_MODEL),
        base_url=os.getenv("LLM_ESCALATION_BASE_URL", ESCALATION_BASE_URL).rstrip("/"),
        api_key=SecretStr(os.getenv("LLM_ESCALATION_API_KEY", "")),
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


def sanitize_provider_error(exc: BaseException) -> str:
    """Return an exception message with common credentials replaced safely."""
    message = str(exc)
    message = _BEARER_SECRET.sub(r"\1[REDACTED]", message)
    message = _QUERY_SECRET.sub(r"\1[REDACTED]", message)
    message = _QUOTED_FIELD_SECRET.sub(r"\1[REDACTED]", message)
    return _HEADER_SECRET.sub(r"\1[REDACTED]", message)
