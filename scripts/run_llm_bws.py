#!/usr/bin/env python
"""Run best-worst-scaling annotation of Blixen-Bible pairs with an LLM

    python -m scripts.run_llm_bws --backend deepseek    # DeepSeek API only
    python -m scripts.run_llm_bws --backend munin       # local GPU model only
    python -m scripts.run_llm_bws --backend both        # both, one after the other
    python -m scripts.run_llm_bws --toy                 # save the first 2 prompts
    python -m scripts.run_llm_bws --toy 5               # save the first 5 prompts
    python -m scripts.run_llm_bws --limit 10 --overwrite

Reads ANNOTATIONS/bws_tuples.csv (four rows per tuple). For every tuple the LLM
chooses the pair with the most convincing intertextual relationship (BEST) and
the least convincing one (WORST), following the best-worst scaling method.

Answers are saved per model under out_dir/<model>/raw and out_dir/<model>/parsed;
a crashed run can simply be restarted (already-answered tuples are skipped
unless --overwrite is set). The aggregate CSV and the input CSV with the picks
filled into the BEST/WORST columns are rebuilt from the parsed files at the end.
DEEPSEEK_API_KEY must be set for --backend deepseek.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bws import (ANNOTATIONS, BACKENDS, DATA, DEEPSEEK_MODEL, MUNIN_MODEL,
                       run_llm_bws)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None,
                        help="tuple CSV to annotate (default: ANNOTATIONS/bws_tuples.csv)")
    parser.add_argument("--out-dir", type=Path, default=DATA / "LLM_annotations")
    parser.add_argument("--backend", choices=(*BACKENDS, "both"), default="both",
                        help="which LLM to use (default: both)")
    parser.add_argument("--munin-model", default=MUNIN_MODEL,
                        help="huggingface name or local directory path of the "
                             "model for --backend munin")
    parser.add_argument("--deepseek-model", default=DEEPSEEK_MODEL,
                        help="model id for --backend deepseek")
    parser.add_argument("--limit", type=int, default=0,
                        help="annotate at most this many pending tuples")
    parser.add_argument("--toy", nargs="?", const=2, type=int, metavar="N",
                        help="save and print the first N prompts without loading "
                             "any model or calling any API (default 2)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-annotate tuples that already have a saved response")
    parser.add_argument("--timeout", type=int, default=120,
                        help="deepseek request timeout in seconds")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="max generated tokens per answer (munin only)")
    args = parser.parse_args()

    backends = BACKENDS if args.backend == "both" else (args.backend,)
    run_llm_bws(args.input, args.out_dir, backends,
                munin_model=args.munin_model,
                deepseek_model=args.deepseek_model,
                limit=args.limit, toy=args.toy, overwrite=args.overwrite,
                timeout=args.timeout, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
