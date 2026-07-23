"""Safe exception-class normalization shared by runtime and evaluation code."""

from __future__ import annotations

import re
from typing import Any

MODEL_GATEWAY_ERROR_CLASSES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "GatewayError",
        "HTTPStatusError",
        "PoolTimeout",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
)

_SAFE_EXCEPTION_CLASS_RE = re.compile(
    r"(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception)|Exception|BaseException)"
)


def normalize_exception_class(value: Any) -> str:
    """Return only a safe exception type name, including gateway timeouts."""
    if isinstance(value, BaseException):
        candidate = type(value).__name__
    else:
        candidate = str(value or "").strip().partition(":")[0].strip()
    if candidate in MODEL_GATEWAY_ERROR_CLASSES:
        return candidate
    if _SAFE_EXCEPTION_CLASS_RE.fullmatch(candidate):
        return candidate
    return ""
