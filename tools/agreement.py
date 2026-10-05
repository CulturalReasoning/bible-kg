"""Inter-annotator agreement for the BWS annotation study

Two human annotators (A, B) and the LLM annotators each mark one BEST and one
WORST position per tuple of four pairs. Agreement is computed per pair of
annotators on the tuples where both made exactly one BEST and one WORST mark:

- Cohen's kappa on the BEST choice and on the WORST choice (4 categories each),
  plus kappa on the combined (best, worst) pair (16 categories), over the
  tuples where both annotators made exactly one BEST and one WORST mark
- the Spearman rank correlation between the two annotators' per-pair
  three-step values: 1 for pairs ranked best, -1 for pairs ranked worst,
  0 for unranked (or tied) pairs

Each pair appears in several tuples (the point of BWS); per annotator we also
count how often a pair was ranked BEST and how often WORST. Each tuple
contributes at most one best vote and one worst vote: if k pairs carry the
mark, each gets 1/k, so an all-worst tuple (e.g. the humans on Q0002) gives
each of its four pairs 0.25 of a worst vote. The score is best - worst.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

LABELS = ["1", "2", "3", "4"]


def read_annotations(path: Path) -> tuple[dict[str, dict[str, set[str]]],
                                           dict[tuple[str, str], str]]:
    """Read an annotation CSV into ({tuple_id: {"best": {positions},
    "worst": {positions}}}, {(tuple_id, position): pair_id}).

    The human sheets start with a comment line that has more fields than the
    header, so the header row is located by its first cell instead of assuming
    it is the first line of the file.
    """
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header_idx = next(
        (i for i, row in enumerate(rows)
         if row and row[0].strip().lower() == "tuple_id"), None)
    if header_idx is None:
        raise ValueError(f"no tuple_id header row found in {path}")
    header = [cell.strip() for cell in rows[header_idx]]

    marks: dict[str, dict[str, set[str]]] = {}
    pairs: dict[tuple[str, str], str] = {}
    for row in rows[header_idx + 1:]:
        if not row or not row[0].strip():
            continue
        record = dict(zip(header, [cell.strip() for cell in row]))
        tuple_id = record.get("tuple_id", "")
        position = record.get("position", "")
        if not tuple_id or not position:
            continue
        entry = marks.setdefault(tuple_id, {"best": set(), "worst": set()})
        pairs[(tuple_id, position)] = record.get("pair_id", "")
        if record.get("BEST", "").lower() == "x":
            entry["best"].add(position)
        if record.get("WORST", "").lower() == "x":
            entry["worst"].add(position)
    return marks, pairs


def check_single_marks(marks: dict[str, dict[str, set[str]]],
                       name: str) -> list[tuple[str, int, int]]:
    """Tuples where the annotator did not mark exactly one BEST and one WORST."""
    issues = []
    for tuple_id in sorted(marks):
        entry = marks[tuple_id]
        n_best, n_worst = len(entry["best"]), len(entry["worst"])
        if n_best != 1 or n_worst != 1:
            issues.append((tuple_id, n_best, n_worst))
    return issues


def _valid_picks(entry: dict[str, set[str]]) -> tuple[str, str] | None:
    if len(entry["best"]) != 1 or len(entry["worst"]) != 1:
        return None
    best, worst = next(iter(entry["best"])), next(iter(entry["worst"]))
    return (best, worst) if best != worst else None


def pair_agreement(marks1: dict[str, dict[str, set[str]]],
                   pairs1: dict[tuple[str, str], str],
                   marks2: dict[str, dict[str, set[str]]],
                   pairs2: dict[tuple[str, str], str]) -> dict:
    """Cohen's kappa on the per-tuple picks and Spearman on the per-pair three-step values."""
    tuple_ids1 = sorted(marks1)
    tuple_ids2 = sorted(marks2)
    assert tuple_ids1 == tuple_ids2, "annotators annotated different tuple sets"

    # kappa: needs exactly one BEST and one WORST per tuple
    best1: list[str] = []
    best2: list[str] = []
    worst1: list[str] = []
    worst2: list[str] = []
    total = partial = disagree = 0
    skipped = 0
    for tuple_id in tuple_ids1:
        picks1 = _valid_picks(marks1[tuple_id])
        picks2 = _valid_picks(marks2[tuple_id])
        if picks1 is None or picks2 is None:   # e.g. the all-worst Q0002
            skipped += 1
            continue
        best1.append(picks1[0])
        best2.append(picks2[0])
        worst1.append(picks1[1])
        worst2.append(picks2[1])
        same = (picks1[0] == picks2[0]) + (picks1[1] == picks2[1])
        if same == 2:
            total += 1
        elif same == 1:
            partial += 1
        else:
            disagree += 1

    # Spearman over the three-step per-pair values (1 best, -1 worst, 0 unranked)
    counts1 = bws_counts(marks1, pairs1)
    counts2 = bws_counts(marks2, pairs2)
    steps1 = pair_steps(counts1)
    steps2 = pair_steps(counts2)
    shared = sorted(set(steps1) & set(steps2))
    rho, pvalue = spearmanr([steps1[pair] for pair in shared],
                            [steps2[pair] for pair in shared])

    result = {
        "n": len(best1),
        "n_skipped": skipped,
        "n_total": total,
        "n_partial": partial,
        "n_disagree": disagree,
        "n_pairs": len(shared),
        "spearman": float(rho),
        "spearman_p": float(pvalue),
    }
    if result["n"]:
        result.update({
            "kappa_best": cohen_kappa_score(best1, best2, labels=LABELS),
            "kappa_worst": cohen_kappa_score(worst1, worst2, labels=LABELS),
            "kappa_pair": cohen_kappa_score(
                [f"{b}:{w}" for b, w in zip(best1, worst1)],
                [f"{b}:{w}" for b, w in zip(best2, worst2)]),
            "agree_best": float(np.mean([a == b for a, b in zip(best1, best2)]) * 100),
            "agree_worst": float(np.mean([a == b for a, b in zip(worst1, worst2)]) * 100),
        })
    return result


def bws_counts(marks: dict[str, dict[str, set[str]]],
               pairs: dict[tuple[str, str], str]) -> dict[str, dict[str, float]]:
    """Times each pair was ranked best / worst, with tied marks split evenly.

    Every pair in the file gets an entry (a pair never marked scores 0/0).
    Each tuple contributes at most one best vote and one worst vote: if k
    pairs carry the mark, each gets 1/k, so an all-worst tuple gives each of
    its four pairs 0.25 of a worst vote.
    """
    counts = {pair_id: {"best": 0.0, "worst": 0.0}
              for pair_id in set(pairs.values()) if pair_id}
    for tuple_id, entry in marks.items():
        for column in ("best", "worst"):
            marked = entry[column]
            if not marked:
                continue
            share = 1.0 / len(marked)
            for position in marked:
                pair_id = pairs.get((tuple_id, position), "")
                if pair_id:
                    counts[pair_id][column] += share
    return counts


def pair_steps(counts: dict[str, dict[str, float]]) -> dict[str, int]:
    """The three-step per-pair value: 1 (more best than worst votes),
    -1 (more worst), 0 (unranked or tied)."""
    steps = {}
    for pair_id, entry in counts.items():
        if entry["best"] > entry["worst"]:
            steps[pair_id] = 1
        elif entry["worst"] > entry["best"]:
            steps[pair_id] = -1
        else:
            steps[pair_id] = 0
    return steps


def short_name(path: Path) -> str:
    """A readable annotator label: the human's first name or the LLM's dir name."""
    if "alexander" in path.name.lower():
        return "Alexander"
    if "jens" in path.name.lower():
        return "Jens"
    return path.parent.name


def run_agreement(human_a: Path, human_b: Path,
                  llm_paths: Iterable[Path]) -> None:
    """Print the multi-mark check, the BWS counts, and the pairwise agreement table."""
    paths = [human_a, human_b, *llm_paths]
    annotators: dict[str, tuple[dict[str, dict[str, set[str]]],
                                 dict[tuple[str, str], str]]] = {}
    for path in paths:
        annotators[short_name(path)] = read_annotations(path)

    print("multi-mark check (tuples without exactly one BEST and one WORST)")
    print("-" * 72)
    for name, (marks, _) in annotators.items():
        issues = check_single_marks(marks, name)
        if issues:
            for tuple_id, _, _ in issues:
                entry = marks[tuple_id]
                print(f"{name:12s} {tuple_id}: BEST={sorted(entry['best'])} "
                      f"WORST={sorted(entry['worst'])}")

    print(f"\nagreement on shared tuples")
    print("-" * 72)
    header = (f"{'pair':34s} {'k_best':>7s} {'k_worst':>7s} {'rho':>7s} "
              f"{'total':>7s} {'partial':>9s} {'disagree':>9s}")
    print(header)
    print("-" * len(header))
    pairs = [(human_a, human_b)] + \
            [(path, human_a) for path in llm_paths] + \
            [(path, human_b) for path in llm_paths]

    pairs.append(tuple([p for p in llm_paths]))

    for left, right in pairs:
        agreement = pair_agreement(annotators[short_name(left)][0],
                                   annotators[short_name(left)][1],
                                   annotators[short_name(right)][0],
                                   annotators[short_name(right)][1])
        label = f"{short_name(left)} vs {short_name(right)}"
        if agreement["n"] == 0:
            print(f"{label:34s} no tuples with single BEST/WORST picks")
            continue
        print(f"{label:34s} {agreement['kappa_best']:>7.3f} "
              f"{agreement['kappa_worst']:>7.3f} "
              f"{agreement['spearman']:>7.3f} "
              f"{agreement['n_total']:>7d} {agreement['n_partial']:>9d} "
              f"{agreement['n_disagree']:>9d}")
    print("\ntotal/partial/disagree = tuples where both annotators made exactly "
          "one BEST and one WORST and both / exactly one / none of the two picks "
          "match (Q0002 excluded for the humans); rho is Spearman over the "
          "per-pair three-step values")


    print("\nbest/worst counts per pair (score = best - worst)")
    print("a tuple with k marks in a column splits one vote among them: "
          "an all-worst tuple gives each pair 0.25")
    print("-" * 72)

