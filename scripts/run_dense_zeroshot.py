#!/usr/bin/env python
"""Evaluate zero-shot dense retrieval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import Corpus, GoldSet
from tools.dense import DenseRetriever
from tools.metrics import by_band, comparison_table

DEFAULT_MODELS = [
    "intfloat/multilingual-e5-large",
    "KennethEnevoldsen/dfm-sentence-encoder-large",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "KennethEnevoldsen/dfm-sentence-encoder-small",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-ortho", action="store_true",
                        help="skip the aa->å orthographic normalisation")
    parser.add_argument("--clean-queries", action="store_true",
                        help="also strip dialogue/attribution from the Blixen passages")
    parser.add_argument("--ablate-ortho", action="store_true",
                        help="run each model both with and without ortho normalisation")
    parser.add_argument("--bands", action="store_true",
                        help="also print the per-relation-band table for each run")
    args = parser.parse_args()

    corpus = Corpus.load()
    gold = GoldSet.load(corpus)
    print(f"corpus {len(corpus)} verses | gold {len(gold)} references "
          f"| bands {gold.band_counts()}\n")

    ortho_settings = [True, False] if args.ablate_ortho else [not args.no_ortho]

    results = {}
    for model_name in args.models:
        for use_ortho in ortho_settings:
            label = model_name.split("/")[-1] + ("__ortho" if use_ortho else "__noortho")
            print(f"{label} ...", flush=True)
            started = time.time()
            retriever = DenseRetriever(model_name, use_ortho=use_ortho,
                                       clean_queries=args.clean_queries,
                                       device=args.device, batch_size=args.batch_size)
            retriever.encode_corpus(corpus.documents)
            results[label] = retriever.ranks(gold)
            print(f"  done in {time.time() - started:.0f}s")
            if args.bands:
                print(by_band(gold, results[label]).round(3).to_string())
            del retriever

    print("\n=== overall (sorted by R@10) ===")
    print(comparison_table(results).round(3).to_string())


if __name__ == "__main__":
    main()