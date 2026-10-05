#!/usr/bin/env python
"""Inter-annotator agreement for the BWS annotation study

    python -m scripts.run_agreement

Computes, per pair of annotators, Cohen's kappa on the BEST and WORST choices
(and on the combined choice) over the shared tuples, and the Spearman rank
correlation between the annotators' per-pair three-step values: 1 for pairs
ranked best, -1 for pairs ranked worst, 0 for unranked (or tied) pairs.
Defaults to the two human sheets in ANNOTATIONS and every LLM annotated CSV
under data/LLM_annotations; override with --human-a/--human-b/--llm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.agreement import run_agreement
from tools.paths import ANNOTATIONS, DATA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-a", type=Path,
                        default=ANNOTATIONS / "bws_tuples_annotator_Alexander.csv")
    parser.add_argument("--human-b", type=Path,
                        default=ANNOTATIONS / "bws_tuples_annotator_Jens.csv")
    parser.add_argument("--llm", nargs="*", type=Path, default=None,
                        help="LLM annotated CSVs (default: every "
                             "data/LLM_annotations/*/bws_tuples_annotated.csv)")
    args = parser.parse_args()

    llm_paths = args.llm
    if llm_paths is None:
        llm_paths = sorted((DATA / "LLM_annotations").glob("*/bws_tuples_annotated.csv"))
    if not llm_paths:
        sys.exit("no LLM annotation files found - pass them with --llm")

    run_agreement(args.human_a, args.human_b, llm_paths)


if __name__ == "__main__":
    main()
