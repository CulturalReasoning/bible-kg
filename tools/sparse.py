"""sparse lexical retrieval: TF-IDF and BM25
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Retriever(Protocol):
    def search(self, tokens: Sequence[str], k: int = 100) -> list[int]:
        ...


def _identity(x):
    return x


class TfidfRetriever:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(analyzer=_identity, sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(corpus_tokens)

    def search(self, tokens: Sequence[str], k: int = 100) -> list[int]:
        from sklearn.metrics.pairwise import linear_kernel

        scores = linear_kernel(self._vectorizer.transform([tokens]), self._matrix).ravel()
        return [int(i) for i in np.argsort(-scores)[:k]]

    def scores(self, tokens: Sequence[str]) -> np.ndarray:
        from sklearn.metrics.pairwise import linear_kernel

        return linear_kernel(self._vectorizer.transform([tokens]), self._matrix).ravel()


class BM25Retriever:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]):
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([list(t) for t in corpus_tokens])

    def search(self, tokens: Sequence[str], k: int = 100) -> list[int]:
        return [int(i) for i in np.argsort(-self._bm25.get_scores(list(tokens)))[:k]]

    def scores(self, tokens: Sequence[str]) -> np.ndarray:
        return self._bm25.get_scores(list(tokens))

    def band(self, tokens: Sequence[str], start: int, stop: int) -> list[int]:
        return [int(i) for i in np.argsort(-self._bm25.get_scores(list(tokens)))[start:stop]]


def rank_all(retriever: Retriever, query_tokens: Sequence[Sequence[str]],
             references, depth: int = 100) -> list[int | None]:
    from .metrics import first_gold_rank

    return [first_gold_rank(retriever.search(tokens, depth), ref.gold)
            for tokens, ref in zip(query_tokens, references)]