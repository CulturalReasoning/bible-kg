"""The gold evaluation set: 189 Blixen-Bible references

from the scholarly commentary to *Syv fantastiske Fortællinger*
was parsed into 164 annotations; annotations citing several distinct references
were split into one row each, giving 189 references

relation: `quote`, `paraphrase` or `allusion` - computed by Jaccard overlap of DaCy lemmas between the Blixen
passage and the verse
"""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from .corpus import Corpus
from .paths import RELATION_JACCARD, SPLIT_ANNOTATIONS, annotation_file

BANDS = ("quote", "paraphrase", "allusion")


@dataclass
class Reference:
    id: str
    annotation_id: str          #: groups references sharing one commentary entry
    text: str                   #: the Blixen passage — the retrieval query
    gold: set[int]              #: corpus indices of the cited verse(s)
    band: str                   #: quote | paraphrase | allusion
    jaccard: float              #: lemma overlap that produced the band
    reference: str              #: human-readable citation, e.g. "Esajas' Bog 66:24"
    story: str = ""
    page: int | None = None
    verse_type: str = ""        #: "single" or "range"
    sibling_gold: set[int] = field(default_factory=set)

    def __repr__(self) -> str:
        return f"Reference({self.id!r}, band={self.band}, |gold|={len(self.gold)})"


class GoldSet(Sequence[Reference]):
    def __init__(self, references: list[Reference]):
        self._refs = references
        self._by_id = {r.id: r for r in references}

    @classmethod
    def load(cls, corpus: Corpus) -> "GoldSet":
        split = json.loads(annotation_file(SPLIT_ANNOTATIONS).read_text(encoding="utf-8"))
        bands = {r["id"]: r for r in
                 json.loads(annotation_file(RELATION_JACCARD).read_text(encoding="utf-8"))}

        references: list[Reference] = []
        by_annotation: dict[str, set[int]] = collections.defaultdict(set)
        for row in split:
            gold = corpus.span(row["book"], row["chapter"],
                               row["verse"], row["verse_end"] or row["verse"])
            if not gold:
                continue
            band = bands[row["id"]]
            references.append(Reference(
                id=row["id"],
                annotation_id=row["annotation_id"],
                text=row["quote_from_work"],
                gold=gold,
                band=band["jaccard_group"],
                jaccard=band["jaccard"],
                reference=band["reference"],
                story=row.get("story", ""),
                page=row.get("page"),
                verse_type=row.get("verse_type", ""),
            ))
            by_annotation[row["annotation_id"]] |= gold

        for ref in references:             
            ref.sibling_gold = by_annotation[ref.annotation_id]
        return cls(references)


    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, index):       
        return self._refs[index]

    def __iter__(self) -> Iterator[Reference]:
        return iter(self._refs)

    def by_id(self, ref_id: str) -> Reference:
        return self._by_id[ref_id]
    

    @property
    def queries(self) -> list[str]:
        return [r.text for r in self._refs]

    def band_counts(self) -> dict[str, int]:
        return {b: sum(r.band == b for r in self._refs) for b in BANDS}

    def band_indices(self) -> dict[str, list[int]]:
        return {b: [i for i, r in enumerate(self._refs) if r.band == b] for b in BANDS}

    def stratified_group_folds(self, n_splits: int = 5, seed: int = 42):
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        x = np.zeros(len(self._refs))
        y = [r.band for r in self._refs]
        groups = [r.annotation_id for r in self._refs]
        for train_idx, test_idx in splitter.split(x, y, groups):
            yield ([self._refs[i] for i in train_idx],
                   [self._refs[i] for i in test_idx])