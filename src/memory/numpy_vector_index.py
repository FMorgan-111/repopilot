"""Portable SQLite + NumPy vector index used when sqlite-vec is unavailable."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

from .vector_index import VectorHit, VectorIndex


class NumpyVectorIndex(VectorIndex):
    """Persist float32 vectors in SQLite and rank them with NumPy cosine distance."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        dim: int,
        table: str = "episode_vectors_numpy",
    ) -> None:
        self.conn = conn
        self.dim = dim
        self.table = table
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )

    def add(self, rowid: int, vector: Sequence[float]) -> None:
        values = self._as_vector(vector)
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self.table}(rowid, embedding) VALUES (?, ?)",
            (rowid, values.tobytes()),
        )

    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        query = self._as_vector(vector)
        if k <= 0:
            return []
        query_norm = float(np.linalg.norm(query))
        hits: list[VectorHit] = []
        for rowid, blob in self.conn.execute(
            f"SELECT rowid, embedding FROM {self.table}"
        ):
            candidate = np.frombuffer(blob, dtype=np.float32)
            if candidate.size != self.dim:
                continue
            candidate_norm = float(np.linalg.norm(candidate))
            if query_norm == 0.0 or candidate_norm == 0.0:
                distance = 1.0
            else:
                similarity = float(np.dot(query, candidate) / (query_norm * candidate_norm))
                distance = 1.0 - max(-1.0, min(1.0, similarity))
            hits.append(VectorHit(rowid=int(rowid), distance=distance))
        hits.sort(key=lambda hit: (hit.distance, hit.rowid))
        return hits[:k]

    def _as_vector(self, vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or values.size != self.dim:
            got = int(values.size)
            raise ValueError(f"expected dim {self.dim}, got {got}")
        return values
