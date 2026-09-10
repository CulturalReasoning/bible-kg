#!/usr/bin/env python
"""Build the best-worst-scaling study: the candidate pool and the annotation tuples

    python scripts/build_bws.py
    python scripts/build_bws.py --out-dir artifacts/bws --seed 20260906

See `tools/bws.py` for details
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import Corpus, GoldSet
from tools.bws import DEFAULT_SEED, REPS, build_pool, build_tuples, write_pool, write_study
from tools.paths import ARTIFACTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "bws")
    parser.add_argument("--reps", type=int, default=REPS,
                        help="how many tuples each pair appears in")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    corpus = Corpus.load()
    gold = GoldSet.load(corpus)

    pool = build_pool(corpus, gold)
    counts = collections.Counter(row["bin"] for row in pool)
    print(f"pool: {len(pool)} pairs from {len(gold)} references")
    for bin_name in sorted(counts):
        print(f"  {bin_name:20s} {counts[bin_name]:5d}")
    print(f"wrote {write_pool(pool, args.out_dir)}")

    tuples, dropped = build_tuples(pool, reps=args.reps, seed=args.seed)
    annotator, key = write_study(pool, tuples, args.out_dir)
    print(f"\ntuples: {len(tuples)} of {args.reps} reps "
          f"({dropped} couldn't be placed into tuples)")
    print(f"wrote {annotator}\nwrote {key}")


if __name__ == "__main__":
    main()