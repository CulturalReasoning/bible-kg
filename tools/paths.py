from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# DATA is for the Bible verses
# ANNOTATIONS is for Blixen's work
DATA = Path(os.environ.get("CURE_DATA", ROOT / "data")).resolve()
ANNOTATIONS = Path(os.environ.get(
    "CURE_ANNOTATIONS", ROOT.parent / "data-annotations")).resolve()

EMBEDDINGS = Path(os.environ.get("CURE_EMBEDDINGS", ROOT / "embeddings")).resolve()
ARTIFACTS = Path(os.environ.get("CURE_ARTIFACTS", ROOT / "artifacts")).resolve()

VERSES = "all_verses_ot_nt.json"              # 31,170 Bible verses (GT1871 + NT1907)

# keep these in the annotations folder
SPLIT_ANNOTATIONS = "split-annotations.json" 
RELATION_JACCARD = "relation_labeling_jaccard.json"   # quote/paraphrase/allusion categories
DACY_TOKENS = "retrieval_tokens_dacy_small.json"      # precomputed lemma+stem tokens
RANKS_ANNOTATED = "retrieval-ranks_annotated.csv"     # expert rank inspection sheet
PER_QUERY_OOF = "per_query_oof.json"                  # fine-tuned out-of-fold rankings

RESTRICTED = (SPLIT_ANNOTATIONS, RELATION_JACCARD, DACY_TOKENS,
              RANKS_ANNOTATED, PER_QUERY_OOF)


def data_file(name: str) -> Path:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found in {DATA}.")
    return path


def annotation_file(name: str) -> Path:
    path = ANNOTATIONS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found in {ANNOTATIONS}.\n"
        )
    return path


def annotations_available() -> bool:
    return all((ANNOTATIONS / name).exists() for name in RESTRICTED)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path