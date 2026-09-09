#!/usr/bin/env python
"""Fine-tune a Danish sentence encoder on the gold references, cross-validated, grouped, stratified 5-fold, checkpoint after every fold

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure import Corpus, GoldSet
from cure.finetune import FinetuneConfig, cross_validate
from cure.metrics import METRICS
from cure.paths import ARTIFACTS, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = FinetuneConfig()
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--folds", type=int, default=defaults.folds)
    parser.add_argument("--seeds", type=int, default=defaults.n_seeds,
                        help="models averaged into the weight soup per fold")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--pretrain", choices=["none", "tsdae", "simcse"],
                        default=defaults.pretrain,
                        help="unsupervised domain adaptation before the supervised step")
    parser.add_argument("--no-ortho", action="store_true",
                        help="skip the aa->å normalisation (ablation only)")
    parser.add_argument("--no-clean-queries", action="store_true",
                        help="keep dialogue delimiters and attribution in queries")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()

    config = FinetuneConfig(
        model=args.model, folds=args.folds, n_seeds=args.seeds, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr, pretrain=args.pretrain,
        use_ortho=not args.no_ortho, clean_queries=not args.no_clean_queries,
        device=args.device,
    )
    out_dir = ensure_dir(args.out_dir)
    print(config.describe(), "\n")

    corpus = Corpus.load()
    gold = GoldSet.load(corpus)
    results = cross_validate(corpus, gold, config,
                             checkpoint_path=out_dir / "cv_checkpoint.json")

    print("\n=== cross-validated ===")
    for metric in METRICS:
        row = results["summary"][metric]
        print(f"  {metric:8s} {row['zero_shot']:.3f} -> {row['finetuned']:.3f} "
              f"± {row['finetuned_std']:.3f}  (Δ {row['finetuned'] - row['zero_shot']:+.3f})")

    print("\n=== fine-tuned (mean over folds) ===")
    from cure.gold import BANDS
    header = "  {:12s}".format("band") + "".join(f"{m:>10s}" for m in METRICS)
    print(header)
    for band in BANDS:
        import numpy as np
        means = {m: float(np.nanmean([f["finetuned"][band][m] for f in results["folds"]]))
                 for m in METRICS}
        print("  {:12s}".format(band) + "".join(f"{means[m]:10.3f}" for m in METRICS))

    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()