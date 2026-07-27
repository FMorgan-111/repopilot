"""RepoPilot v2 agent graph node functions."""

from typing import Any

__all__ = ["ensure_coverage"]


def __getattr__(name: str) -> Any:
    if name == "ensure_coverage":
        from .coverage import ensure_coverage

        return ensure_coverage
    raise AttributeError(name)
