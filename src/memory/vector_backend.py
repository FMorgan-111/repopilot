"""Select the fastest available vector-index backend."""

from __future__ import annotations

import sqlite3
import warnings

from .vector_index import VectorIndex


def _create_sqlite_vec_index(conn: sqlite3.Connection, dim: int) -> VectorIndex:
    """Import the optional sqlite-vec backend only when memory is enabled."""
    from .sqlite_vec_index import SqliteVecIndex

    return SqliteVecIndex(conn, dim)


def _create_numpy_vector_index(conn: sqlite3.Connection, dim: int) -> VectorIndex:
    """Import the optional NumPy fallback only after sqlite-vec is unavailable."""
    from .numpy_vector_index import NumpyVectorIndex

    return NumpyVectorIndex(conn, dim)

_fallback_warning_emitted = False


def create_vector_index(
    conn: sqlite3.Connection, dim: int
) -> tuple[VectorIndex, str]:
    """Prefer sqlite-vec and visibly fall back on unsupported Python builds."""
    try:
        return _create_sqlite_vec_index(conn, dim), "sqlite_vec"
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
        return _create_numpy_vector_index(conn, dim), "numpy"
