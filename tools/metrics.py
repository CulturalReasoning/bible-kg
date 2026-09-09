"""Retrieval metrics: P@1, R@10, MRR@10, NDCG@10
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

METRICS = ("P@1", "R@10", "MRR@10", "NDCG@10")

K = 10


def first_gold_rank(ranked: Sequence[int], gold: set[int]) -> int | None:
    return next((i + 1 for i, doc in enumerate(ranked) if doc in gold), None)


def _dcg(relevances: Iterable[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_from_ranking(ranked: Sequence[int], gold: set[int], k: int = K) -> float:
    relevances = [1 if doc in gold else 0 for doc in ranked[:k]]
    ideal = _dcg([1] * min(len(gold), k))
    return _dcg(relevances) / ideal if ideal > 0 else 0.0


def ndcg_from_rank(rank: int | None, k: int = K) -> float:
    return 1.0 / math.log2(rank + 1) if rank and rank <= k else 0.0


def metrics_from_ranking(ranked: Sequence[int], gold: set[int], k: int = K) -> dict[str, float]:
    rank = first_gold_rank(ranked, gold)
    return {
        "P@1": 1.0 if ranked and ranked[0] in gold else 0.0,
        "R@10": 1.0 if rank and rank <= k else 0.0,
        "MRR@10": 1.0 / rank if rank and rank <= k else 0.0,
        "NDCG@10": ndcg_from_ranking(ranked, gold, k),
    }


def metrics_from_ranks(ranks: Iterable[int | None], k: int = K) -> dict[str, float]:
    ranks = [r if r else 10**9 for r in ranks]
    if not ranks:
        return {metric: float("nan") for metric in METRICS}
    return {
        "P@1": float(np.mean([r == 1 for r in ranks])),
        "R@10": float(np.mean([r <= k for r in ranks])),
        "MRR@10": float(np.mean([1.0 / r if r <= k else 0.0 for r in ranks])),
        "NDCG@10": float(np.mean([ndcg_from_rank(r, k) for r in ranks])),
    }


def by_band(references, ranks: Sequence[int | None], k: int = K) -> "pd.DataFrame":
    import pandas as pd

    from .gold import BANDS

    rows, index = [], []
    for band in BANDS:
        selected = [ranks[i] for i, r in enumerate(references) if r.band == band]
        rows.append(metrics_from_ranks(selected, k))
        index.append(f"{band} (n={len(selected)})")
    rows.append({m: float(np.nanmean([rows[i][m] for i in range(len(BANDS))]))
                 for m in METRICS})
    index.append("macro-avg")
    rows.append(metrics_from_ranks(ranks, k))
    index.append(f"overall (n={len(list(ranks))})")
    return pd.DataFrame(rows, index=index, columns=list(METRICS))


def comparison_table(results: dict[str, Sequence[int | None]], k: int = K) -> "pd.DataFrame":
    import pandas as pd

    table = pd.DataFrame({name: metrics_from_ranks(ranks, k)
                          for name, ranks in results.items()}).T
    return table[list(METRICS)].sort_values("R@10", ascending=False)