"""Best-worst scaling: tuples for expert/LLM annotation

"""
from __future__ import annotations

import collections
import csv
import json
import random
from pathlib import Path

import numpy as np

from .corpus import Corpus
from .gold import GoldSet
from .paths import PER_QUERY_OOF, annotation_file
from .sparse import BM25Retriever
from .text import jaccard, load_dacy_tokens

TOP_K = 3            #: candidates taken from each retriever
REPS = 4             #: tuples each pair appears in
TUPLE_SIZE = 4
MAX_PER_ANNOTATION = 2   #: pairs sharing one Blixen passage inside a tuple
MAX_PER_BIN = 2          #: pairs from one bin inside a tuple
DEFAULT_SEED = 20260906

BINS = ("1_both", "2_bm25_only", "3_dfmft_only_jac", "4_dfmft_only_zero")


def load_finetuned_rankings() -> dict[str, list[int]]:
    rows = json.loads(annotation_file(PER_QUERY_OOF).read_text(encoding="utf-8"))
    return {row["id"]: row["ft_oof_top20"] for row in rows}


def build_pool(corpus: Corpus, gold: GoldSet) -> list[dict]:
    tokens = load_dacy_tokens()
    tokens.check_alignment(len(corpus))
    corpus_tokens = [set(t) for t in tokens.corpus_stem]
    bm25 = BM25Retriever(tokens.corpus_stem)
    finetuned = load_finetuned_rankings()

    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for reference in gold:
        query_tokens = sorted(set(tokens.query_stem[reference.id]))
        bm25_ranks = {doc: rank for rank, doc
                      in enumerate(bm25.search(query_tokens, 200), start=1)}
        ft_ranks = {int(doc): rank for rank, doc
                    in enumerate(finetuned[reference.id], start=1)}

        excluded = (corpus.duplicates_of(reference.gold)
                    | _adjacent_verses(corpus, reference))
        bm25_top = {d for d in bm25.search(query_tokens, TOP_K) if d not in excluded}
        ft_top = {int(d) for d in finetuned[reference.id][:TOP_K] if int(d) not in excluded}

        for candidate in sorted(bm25_top | ft_top):
            if (reference.id, candidate) in seen:
                continue
            seen.add((reference.id, candidate))
            overlap = jaccard(query_tokens, corpus_tokens[candidate])
            verse = corpus[candidate]
            rows.append({
                "pair_id": f"{reference.id}__{verse.book}{verse.chapter}.{verse.verse}",
                "annotation_id": reference.id,
                "reference": corpus.reference(candidate),
                "category": reference.band,
                "bin": _bin_for(candidate, bm25_top, ft_top, overlap),
                "jaccard": f"{overlap:.3f}",
                "blixen_text": reference.text,
                "bible_text": verse.text,
                "bm25_lemstem_rank": bm25_ranks.get(candidate, ""),
                "dfm_ft_rank": ft_ranks.get(candidate, ""),
            })

    rows.sort(key=lambda r: (r["bin"], r["annotation_id"], r["reference"]))
    return rows


def _adjacent_verses(corpus: Corpus, reference) -> set[int]:
    ordered = sorted(reference.gold)
    first, last = corpus[ordered[0]], corpus[ordered[-1]]
    neighbours = (corpus.index_of(first.book, first.chapter, first.verse - 1),
                  corpus.index_of(last.book, last.chapter, last.verse + 1))
    return {i for i in neighbours if i is not None}


def _bin_for(candidate: int, bm25_top: set[int], ft_top: set[int], overlap: float) -> str:
    if candidate in bm25_top and candidate in ft_top:
        return "1_both"
    if candidate in bm25_top:
        return "2_bm25_only"
    return "3_dfmft_only_jac" if overlap > 0 else "4_dfmft_only_zero"


def build_tuples(pool: list[dict], reps: int = REPS, size: int = TUPLE_SIZE,
                 seed: int = DEFAULT_SEED) -> tuple[list[list[str]], int]:
    """Assign pool rows into 4-tuples. Returns (tuples, dropped slots).

    greedy, fill each tuple from the largest bin that has not hit its cap, skipping
    pairs already in the tuple or over the per-annotation cap
    """
    rng = random.Random(seed)
    by_bin: dict[str, list[str]] = collections.defaultdict(list)
    for row in pool:
        by_bin[row["bin"]].extend([row["pair_id"]] * reps)
    for bin_name in by_bin:
        rng.shuffle(by_bin[bin_name])

    info = {row["pair_id"]: row for row in pool}
    tuples: list[list[str]] = []

    while sum(len(v) for v in by_bin.values()) >= size:
        chosen: list[str] = []
        per_annotation: collections.Counter = collections.Counter()
        per_bin: collections.Counter = collections.Counter()

        while len(chosen) < size:
            available = [b for b in by_bin if by_bin[b] and per_bin[b] < MAX_PER_BIN]
            placed = False
            for bin_name in sorted(available, key=lambda b: -len(by_bin[b])):
                for position, pair_id in enumerate(by_bin[bin_name]):
                    annotation = info[pair_id]["annotation_id"]
                    if pair_id in chosen or per_annotation[annotation] >= MAX_PER_ANNOTATION:
                        continue
                    chosen.append(by_bin[bin_name].pop(position))
                    per_annotation[annotation] += 1
                    per_bin[bin_name] += 1
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                break

        if len(chosen) == size:
            rng.shuffle(chosen)
            tuples.append(chosen)
        else:
            for pair_id in chosen:                # return unusable slots and stop
                by_bin[info[pair_id]["bin"]].append(pair_id)
            break

    return tuples, sum(len(v) for v in by_bin.values())


def write_study(pool: list[dict], tuples: list[list[str]], out_dir: Path) -> tuple[Path, Path]:
    info = {row["pair_id"]: row for row in pool}
    annotator_rows, key_rows = [], []
    for number, tuple_pairs in enumerate(tuples, start=1):
        tuple_id = f"T{number:04d}"
        for position, pair_id in enumerate(tuple_pairs, start=1):
            row = info[pair_id]
            annotator_rows.append({
                "tuple_id": tuple_id, "position": position, "pair_id": pair_id,
                "reference": row["reference"], "blixen_text": row["blixen_text"],
                "bible_text": row["bible_text"], "BEST": "", "WORST": "",
            })
            key_rows.append({
                "tuple_id": tuple_id, "position": position, "pair_id": pair_id,
                "annotation_id": row["annotation_id"], "reference": row["reference"],
                "category": row["category"], "bin": row["bin"], "jaccard": row["jaccard"],
                "bm25_lemstem_rank": row["bm25_lemstem_rank"],
                "dfm_ft_rank": row["dfm_ft_rank"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in (("bws_tuples_annotator.csv", annotator_rows),
                       ("bws_tuples_key.csv", key_rows)):
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths[0], paths[1]


def write_pool(pool: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bws_annotation_pool.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pool[0].keys()))
        writer.writeheader()
        writer.writerows(pool)
    return path