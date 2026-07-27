"""sqlite-vec backed vector index — brute-force cosine kNN.

Uses a ``vec0`` virtual table with ``distance_metric=cosine`` so the ``distance``
column is cosine distance (0 = identical, 2 = opposite). At a few thousand rows
the full scan is single-digit milliseconds and needs zero index maintenance,
which is why we prefer it over FAISS here.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Sequence

import sqlite_vec

from .vector_index import VectorHit, VectorIndex


def serialize_f32(vector: Sequence[float]) -> bytes:
    """Pack a float sequence into the compact byte layout sqlite-vec expects."""
    return struct.pack(f"{len(vector)}f", *vector)


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into an existing connection."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


class SqliteVecIndex(VectorIndex):
    def __init__(
        self,
        conn: sqlite3.Connection,
        dim: int,
        table: str = "episode_vectors",
    ) -> None:
        self.conn = conn
        self.dim = dim
        self.table = table
        load_sqlite_vec(conn)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )

    def add(self, rowid: int, vector: Sequence[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {len(vector)}")
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self.table}(rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_f32(vector)),
        )

    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        if len(vector) != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {len(vector)}")
        rows = self.conn.execute(
            f"SELECT rowid, distance FROM {self.table} "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (serialize_f32(vector), k),
        ).fetchall()
        return [VectorHit(rowid=int(r[0]), distance=float(r[1])) for r in rows]
