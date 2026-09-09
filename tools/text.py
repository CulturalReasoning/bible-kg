"""Text normalisation for 19th-century Danish.

orthographic normalisation, query cleaning, lexical tokenisation (Dacy-small lemmas + Snowball stems)
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable

from .paths import DACY_TOKENS, annotation_file


_BOILERPLATE = re.compile(r"Status for dette kapitel.*$", re.S)


def ortho(text: str | None) -> str:
    """NFC-normalise, remove scraper artifact, map archaic `aa` to `å`

    >>> ortho("Han raabte paa Gud.")
    'Han råbte på Gud.'
    """
    text = _BOILERPLATE.sub("", text or "")
    text = unicodedata.normalize("NFC", text).strip()
    return text.replace("aa", "å").replace("Aa", "Å")


_QUOTE_CHARS = "«»“”‘’„‟\"'`"
_SPEECH_VERBS = "sagde|tænkte|spurgte|svarede|raabte|hviskede|udbrød|mente|smilede"
_ATTRIBUTION = re.compile(
    rf",?\s*(?:{_SPEECH_VERBS})\s+(?:han|hun|de|jeg|du|I)\b[^,.;]*", re.I
)


def clean_query(text: str | None) -> str:
    """Strip dialogue punctuation 

    >>> clean_query('»Mange Vande«, sagde han til dem, kan ikke slukke')
    'Mange Vande, kan ikke slukke'
    """
    text = "".join(c for c in (text or "") if c not in _QUOTE_CHARS)
    text = _ATTRIBUTION.sub("", text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def prepare_document(text: str, use_ortho: bool = True) -> str:
    return ortho(text) if use_ortho else text


def prepare_query(text: str, use_ortho: bool = True, clean: bool = True) -> str:
    if clean:
        text = clean_query(text)
    return ortho(text) if use_ortho else text


STOPWORDS = frozenset(
    "og i at det en den til er som på de med han af for ikke der var et har om vi "
    "min så jeg du men kan vil sig nu hans hun være have over efter ved mod eller "
    "fra op ud da kun man mig dig os jer dem sin sit deres vor vore denne dette "
    "disse hvor hvad hvem hvilken alle alt ingen nogen noget meget mere mest jo nej "
    "ja her hen ham hende the saa skal vilde skulde havde blev ere vare thi dog "
    "".split()
)

_SUFFIXES = ("ernes", "erne", "ende", "enes", "ene", "ens", "ets",
             "en", "et", "er", "es", "e", "s", "t")

_WORD = re.compile(r"\w+", re.UNICODE)


def stem(word: str) -> str:
    """Strip Danish suffixes

    >>> stem("Himmelen"), stem("hus")
    ('Himmel', 'hus')
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower())
            if w not in STOPWORDS and len(w) > 1]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class DacyTokens:
    """Precomputed lemma and stem tokens for the corpus and the gold queries

    computed by `precompute_dacy_tokens.py` with 
    DaCy-small lemmatisation, `aa->å` lemma normalisation, stopword and length-2 filtering, then Snowball-Danish stemming
    """

    def __init__(self, payload: dict):
        self.model = payload["model"]
        self.stemmer = payload["stem"]
        self.corpus_lemma = [s.split() for s in payload["corpus_lemma"]]
        self.corpus_stem = [s.split() for s in payload["corpus_stem"]]
        self.query_lemma = {k: v["lemma"].split() for k, v in payload["queries"].items()}
        self.query_stem = {k: v["stem"].split() for k, v in payload["queries"].items()}

    def check_alignment(self, n_verses: int) -> None:
        if len(self.corpus_stem) != n_verses:
            raise ValueError(
                f"DaCy token cache holds {len(self.corpus_stem)} documents but the "
                f"corpus has {n_verses} verses. Please regenerate the cache."
            )

    def __repr__(self) -> str:
        return (f"DacyTokens({self.model} + {self.stemmer}, "
                f"{len(self.corpus_stem)} docs, {len(self.query_stem)} queries)")


def load_dacy_tokens() -> DacyTokens:
    payload = json.loads(annotation_file(DACY_TOKENS).read_text(encoding="utf-8"))
    return DacyTokens(payload)