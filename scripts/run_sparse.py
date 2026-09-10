#!/usr/bin/env python
"""Evaluate lexical retrieval (TF-IDF and BM25)

Runs the four sparse retrievers  {TF-IDF, BM25} two versions for each {raw tokens, DaCy lemma + Snowball stem} and reports the selected four metrics

    python scripts/run_sparse.py
    python scripts/run_sparse.py --retrievers bm25_lemstem --depth 200
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import Corpus, GoldSet
from tools.metrics import by_band, comparison_table
from tools.sparse import BM25Retriever, TfidfRetriever, rank_all
from tools.text import load_dacy_tokens

WORD = re.compile(r"\b\w\w+\b", re.UNICODE)


def raw_tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrievers", nargs="+",
                        default=["tfidf_raw", "bm25_raw", "tfidf_lemstem", "bm25_lemstem"],
                        help="which of the four configurations to run")
    parser.add_argument("--depth", type=int, default=100,
                        help="retrieval depth; metrics are still reported at k=10")
    parser.add_argument("--bands", action="store_true",
                        help="also print the per-relation-band table for each retriever")
    args = parser.parse_args()

    corpus = Corpus.load()
    gold = GoldSet.load(corpus)
    tokens = load_dacy_tokens()
    tokens.check_alignment(len(corpus))
    print(f"corpus {len(corpus)} verses | gold {len(gold)} references "
          f"| bands {gold.band_counts()}")
    print(f"lemma cache: {tokens.model} + {tokens.stemmer}\n")

    corpus_raw = [raw_tokens(d) for d in corpus.documents]
    query_raw = [raw_tokens(r.text) for r in gold]
    query_lemstem = [tokens.query_stem[r.id] for r in gold]

    builders = {
        "tfidf_raw": (TfidfRetriever, corpus_raw, query_raw),
        "bm25_raw": (BM25Retriever, corpus_raw, query_raw),
        "tfidf_lemstem": (TfidfRetriever, tokens.corpus_stem, query_lemstem),
        "bm25_lemstem": (BM25Retriever, tokens.corpus_stem, query_lemstem),
    }

    results = {}
    for name in args.retrievers:
        if name not in builders:
            parser.error(f"unknown retriever {name!r}; choose from {list(builders)}")
        cls, corpus_tokens, query_tokens = builders[name]
        started = time.time()
        ranks = rank_all(cls(corpus_tokens), query_tokens, gold, args.depth)
        results[name] = ranks
        print(f"  {name:15s} done in {time.time() - started:5.1f}s")

    print("\n=== overall ===")
    print(comparison_table(results).round(3).to_string())

    if args.bands:
        for name, ranks in results.items():
            print(f"\n=== {name} by relation band ===")
            print(by_band(gold, ranks).round(3).to_string())


if __name__ == "__main__":
    main()