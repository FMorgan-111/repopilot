"""Traceback keyframe extraction.

A raw pytest failure log can be tens of KB of repeated frames. For semantic
memory we only need a compact, stable fingerprint of *what* failed: the
exception type/message plus the last few stack frames (the ones closest to the
error). This keeps embeddings focused and each stored episode small (<= 2KB).
"""

from __future__ import annotations

import re

# `  File "/path/to/mod.py", line 42, in func_name`
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
# Final `ExceptionType: message` line of a traceback.
_EXC_RE = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Failure)):?(?P<msg>.*)$")

MAX_KEYFRAME_CHARS = 2000
_TOP_FRAMES = 3


def _iter_frames(text: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for raw in text.splitlines():
        match = _FRAME_RE.match(raw)
        if match:
            frames.append(
                {
                    "file": match.group("file"),
                    "line": match.group("line"),
                    "func": match.group("func").strip(),
                }
            )
    return frames


def _exception_line(text: str) -> str:
    """Return the last line that looks like `ExceptionType: message`."""
    candidate = ""
    for raw in text.splitlines():
        line = raw.strip()
        if _EXC_RE.match(line):
            candidate = line
    return candidate


def extract_keyframe(error_log: str) -> str:
    """Compress a traceback / test log to exception type + last N frames.

    Returns a short deterministic string. Falls back to the tail of the log
    when no recognizable traceback structure is present. Always <= 2KB.
    """
    if not error_log:
        return ""

    frames = _iter_frames(error_log)
    exc_line = _exception_line(error_log)

    parts: list[str] = []
    if exc_line:
        parts.append(exc_line)
    # Keep the frames closest to the failure (the last ones in the traceback).
    for frame in frames[-_TOP_FRAMES:]:
        # Drop absolute path noise; the basename + func is the stable signal.
        base = frame["file"].rsplit("/", 1)[-1]
        parts.append(f'{base}:{frame["line"]} in {frame["func"]}')

    if not parts:
        # No traceback structure — use the tail, where errors usually surface.
        parts.append(error_log.strip()[-MAX_KEYFRAME_CHARS:])

    keyframe = "\n".join(parts)
    if len(keyframe) > MAX_KEYFRAME_CHARS:
        keyframe = keyframe[:MAX_KEYFRAME_CHARS].rstrip()
    return keyframe
