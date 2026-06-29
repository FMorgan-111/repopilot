"""Deterministic lexical reranking helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .state import FileInfo

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class BM25Score:
    path: str
    bm25_score: float
    normalized_score: float
    matched_terms: list[str]


def bm25_rerank(query: str, files: list[FileInfo]) -> list[FileInfo]:
    """Return files reranked by BM25 when lexical signal exists."""
    scores = bm25_scores(query, files)
    if not any(score.bm25_score > 0 for score in scores):
        return files

    by_path = {score.path: score for score in scores}
    reranked: list[FileInfo] = []
    for file in files:
        score = by_path[file.path]
        updated = file.model_copy(deep=True)
        if score.bm25_score > 0:
            updated.relevance_score = _blend_score(
                file.relevance_score,
                score.normalized_score,
            )
            matched = ", ".join(score.matched_terms[:6])
            updated.reason = (
                f"{file.reason}; bm25 rerank score={score.bm25_score:.3f}; "
                f"matched issue terms: {matched}"
            ).strip("; ")
        reranked.append(updated)

    return [
        item[1]
        for item in sorted(
            enumerate(reranked),
            key=lambda item: (
                item[1].relevance_score,
                by_path[item[1].path].bm25_score,
                -item[0],
            ),
            reverse=True,
        )
    ]


def bm25_scores(query: str, files: list[FileInfo]) -> list[BM25Score]:
    query_terms = _query_terms(query)
    if not query_terms or not files:
        return [
            BM25Score(
                path=file.path,
                bm25_score=0.0,
                normalized_score=0.0,
                matched_terms=[],
            )
            for file in files
        ]

    documents = [_document_tokens(file) for file in files]
    doc_lengths = [len(document) for document in documents]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    document_frequencies = _document_frequencies(query_terms, documents)
    raw_scores = [
        _bm25_score(
            query_terms,
            document,
            doc_length,
            avg_doc_length,
            document_frequencies,
            len(files),
        )
        for document, doc_length in zip(documents, doc_lengths)
    ]
    max_score = max(raw_scores) if raw_scores else 0.0

    return [
        BM25Score(
            path=file.path,
            bm25_score=score,
            normalized_score=(score / max_score) if max_score > 0 else 0.0,
            matched_terms=_matched_terms(query_terms, documents[index]),
        )
        for index, (file, score) in enumerate(zip(files, raw_scores))
    ]


def _blend_score(existing_score: float, normalized_bm25: float) -> float:
    blended = (existing_score * 0.35) + (normalized_bm25 * 0.65)
    return round(min(max(blended, 0.0), 1.0), 4)


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in _tokenize(query):
        if token in _STOP_WORDS or token in terms:
            continue
        terms.append(token)
    return terms


def _tokenize(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


def _document_tokens(file: FileInfo) -> list[str]:
    if not file.content.strip():
        return []
    path_text = file.path.replace("/", " ").replace("_", " ").replace("-", " ")
    return _tokenize(f"{path_text} {file.content}")


def _document_frequencies(
    query_terms: Iterable[str], documents: list[list[str]]
) -> dict[str, int]:
    return {
        term: sum(1 for document in documents if term in set(document))
        for term in query_terms
    }


def _bm25_score(
    query_terms: list[str],
    document: list[str],
    doc_length: int,
    avg_doc_length: float,
    document_frequencies: dict[str, int],
    document_count: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not document or avg_doc_length <= 0:
        return 0.0

    score = 0.0
    for term in query_terms:
        term_frequency = document.count(term)
        if term_frequency == 0:
            continue
        df = document_frequencies.get(term, 0)
        idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
        denominator = term_frequency + k1 * (
            1 - b + b * (doc_length / avg_doc_length)
        )
        score += idf * ((term_frequency * (k1 + 1)) / denominator)
    return score


def _matched_terms(query_terms: list[str], document: list[str]) -> list[str]:
    document_terms = set(document)
    return [term for term in query_terms if term in document_terms]
