"""Shared paths for RepoPilot's user-owned data."""

from __future__ import annotations

import os
from pathlib import Path


def repopilot_home() -> Path:
    """Return expanded REPOPILOT_HOME or ~/.repopilot."""
    return Path(os.environ.get("REPOPILOT_HOME", "~/.repopilot")).expanduser()
