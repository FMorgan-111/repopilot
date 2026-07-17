"""Select the fastest available vector-index backend."""

from __future__ import annotations

import sqlite3
import warnings

from .numpy_vector_index import NumpyVectorIndex
from .vector_index import VectorIndex

try:
    from .sqlite_vec_index import SqliteVecIndex
except ImportError:  # sqlite-vec is optional
    SqliteVecIndex = None  # type: ignore[assignment,misc]

_fallback_warning_emitted = False


def create_vector_index(
    conn: sqlite3.Connection, dim: int
) -> tuple[VectorIndex, str]:
    """Prefer sqlite-vec and visibly fall back on unsupported Python builds."""
    try:
        if SqliteVecIndex is None:
            raise ImportError("sqlite-vec is not installed")
        return SqliteVecIndex(conn, dim), "sqlite_vec"
    except (ImportError, AttributeError, sqlite3.Error, RuntimeError) as exc:
        global _fallback_warning_emitted
        if not _fallback_warning_emitted:
            warnings.warn(
                "sqlite-vec is unavailable; falling back to NumPy cosine search "
                f"({type(exc).__name__}: {exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            _fallback_warning_emitted = True
        return NumpyVectorIndex(conn, dim), "numpy"
