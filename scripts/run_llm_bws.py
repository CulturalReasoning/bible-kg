#!/usr/bin/env python
"""Run best-worst-scaling annotation with an LLM through the OpenRouter API
python scripts/run_llm_bws.py
python scripts/run_llm_bws.py --limit 10
python scripts/run_llm_bws.py --toy              # save 3 prompts, no API call
python scripts/run_llm_bws.py --overwrite --model anthropic/claude-sonnet-4

I eventually want to make this a loop that fine tune the embedding model at the end of LLM bws annotation, select new candidates, annotate again, fine tune again, and evaluate on the gold set every some runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bws import DEFAULT_MODEL, run_llm_bws
from tools.paths import DATA, ROOT




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "bws_tuples_annotator.csv",
                        help="annotator CSV written by build_bws.py")
    parser.add_argument("--out-dir", type=Path, default=DATA / "LLM_annotations")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0,
                        help="annotate at most this many pending tuples")
    parser.add_argument("--toy", nargs="?", const=3, type=int, metavar="N",
                        help="save and print the first N prompts without any API "
                             "call (default 3)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-annotate tuples that already have a saved response")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    run_llm_bws(args.input, args.out_dir, args.model,
                limit=args.limit, toy=args.toy, overwrite=args.overwrite,
                timeout=args.timeout)


if __name__ == "__main__":
    main()
