"""Dense retrieval methods with sentence-transformers
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .paths import EMBEDDINGS, ensure_dir
from .text import prepare_document, prepare_query


def prefixes_for(model_name: str) -> tuple[str, str]:
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


class DenseRetriever:
    def __init__(self, model_name: str, *, use_ortho: bool = True,
                 clean_queries: bool = False, device: str | None = None,
                 batch_size: int = 64, cache_dir: Path | None = None,
                 model=None):
        self.model_name = model_name
        self.use_ortho = use_ortho
        self.clean_queries = clean_queries
        self.batch_size = batch_size
        self.cache_dir = ensure_dir(Path(cache_dir) if cache_dir else EMBEDDINGS)
        self.query_prefix, self.doc_prefix = prefixes_for(model_name)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
        self.model = model
        self.corpus_embeddings: np.ndarray | None = None

    @classmethod
    def from_model(cls, model, model_name: str, **kwargs) -> "DenseRetriever":
        return cls(model_name, model=model, **kwargs)

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=show_progress).astype("float32")

    def _cache_path(self, n_documents: int) -> Path:
        short = self.model_name.rstrip("/").split("/")[-1]
        tag = "passage" if self.doc_prefix else "noprefix"
        suffix = "_ortho" if self.use_ortho else ""
        return self.cache_dir / f"bible_{short}_{tag}{suffix}.npy"

    def encode_corpus(self, documents: Sequence[str], use_cache: bool = True) -> np.ndarray:
        cache = self._cache_path(len(documents))
        if use_cache and cache.exists():
            embeddings = np.load(cache).astype("float32")
            if embeddings.shape[0] == len(documents):
                self.corpus_embeddings = embeddings
                return embeddings
            print(f"cache {cache.name} is stale "
                  f"({embeddings.shape[0]} != {len(documents)}), re-encoding")

        print(f"encoding {len(documents)} verses with {self.model_name} "
              f"(cached to {cache.name})")
        prepared = [self.doc_prefix + prepare_document(d, self.use_ortho) for d in documents]
        embeddings = self.encode(prepared, show_progress=True)
        if use_cache:
            np.save(cache, embeddings)
        self.corpus_embeddings = embeddings
        return embeddings

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [self.query_prefix + prepare_query(t, self.use_ortho, self.clean_queries)
                    for t in texts]
        return self.encode(prepared)

    def _require_corpus(self) -> np.ndarray:
        if self.corpus_embeddings is None:
            raise RuntimeError("call encode_corpus(...) before searching")
        return self.corpus_embeddings

    def search(self, texts: Sequence[str], k: int = 100) -> list[list[int]]:
        similarities = self.encode_queries(texts) @ self._require_corpus().T
        return [[int(i) for i in np.argsort(-row)[:k]] for row in similarities]

    def ranks(self, references) -> list[int]:
        similarities = self.encode_queries([r.text for r in references]) @ self._require_corpus().T
        return [int(1 + (row > row[list(ref.gold)].max()).sum())
                for row, ref in zip(similarities, references)]

    def ranks_and_top(self, references, k: int = 100) -> tuple[list[int], list[list[int]]]:
        similarities = self.encode_queries([r.text for r in references]) @ self._require_corpus().T
        ranks = [int(1 + (row > row[list(ref.gold)].max()).sum())
                 for row, ref in zip(similarities, references)]
        tops = [[int(i) for i in np.argsort(-row)[:k]] for row in similarities]
        return ranks, tops