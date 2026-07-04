"""Pluggable vector-index abstraction.

The episode store depends on this interface, not on a concrete backend, so the
similarity search can be swapped (sqlite-vec today, FAISS later) without
touching recall/record logic. At the 2k-5k episode scale a brute-force
sqlite-vec scan is 2-5ms, so the default implementation lives in
``sqlite_vec_index.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorHit:
    """A single nearest-neighbour result: episode rowid + distance (cosine)."""

    rowid: int
    distance: float


class VectorIndex(ABC):
    """Maps an integer rowid to a fixed-dimension vector and does kNN search."""

    dim: int

    @abstractmethod
    def add(self, rowid: int, vector: Sequence[float]) -> None:
        """Insert or replace the vector for ``rowid``."""

    @abstractmethod
    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        """Return up to ``k`` nearest neighbours, closest first."""
