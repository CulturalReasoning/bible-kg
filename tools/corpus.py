"""The Bible corpus: 31,170 verses, GT1871 + NT1907.
data files depend on book code keys
"""
from __future__ import annotations

import collections
import json
import unicodedata
from dataclasses import dataclass
from typing import Iterator

from .paths import VERSES, data_file

# Book code: Danish name - for all 66 books of the Protestant canon.
BOOK_NAMES = {
    # Old Testament (39)
    "1mos": "1. Mosebog",
    "2mos": "2. Mosebog",
    "3mos": "3. Mosebog",
    "4mos": "4. Mosebog",
    "5mos": "5. Mosebog",
    "jos": "Josvabogen",
    "dom": "Dommerbogen",
    "ruth": "Ruths Bog",
    "1sam": "1. Samuelsbog",
    "2sam": "2. Samuelsbog",
    "1kong": "1. Kongebog",
    "2kong": "2. Kongebog",
    "1kroen": "1. Krønikebog",
    "2kroen": "2. Krønikebog",
    "ezra": "Ezras Bog",
    "neh": "Nehemias' Bog",
    "est": "Esters Bog",
    "job": "Jobs Bog",
    "sl": "Salmernes Bog",
    "ordsp": "Ordsprogenes Bog",
    "praed": "Prædikerens Bog",
    "hoejs": "Højsangen",
    "es": "Esajas' Bog",
    "jer": "Jeremias' Bog",
    "klages": "Klagesangene",
    "ez": "Ezekiels Bog",
    "dan": "Daniels Bog",
    "hos": "Hoseas' Bog",
    "joel": "Joels Bog",
    "am": "Amos' Bog",
    "obad": "Obadias' Bog",
    "jon": "Jonas' Bog",
    "mika": "Mikas Bog",
    "nah": "Nahums Bog",
    "hab": "Habakkuks Bog",
    "sef": "Sefanias' Bog",
    "hagg": "Haggajs Bog",
    "zak": "Zakarias' Bog",
    "mal": "Malakias' Bog",
    # New Testament (27)
    "matt": "Matthæusevangeliet",
    "mark": "Markusevangeliet",
    "luk": "Lukasevangeliet",
    "joh": "Johannesevangeliet",
    "apg": "Apostlenes Gerninger",
    "rom": "Romerbrevet",
    "1kor": "1. Korintherbrev",
    "2kor": "2. Korintherbrev",
    "gal": "Galaterbrevet",
    "ef": "Efeserbrevet",
    "fil": "Filipperbrevet",
    "kol": "Kolossenserbrevet",
    "1thess": "1. Thessalonikerbrev",
    "2thess": "2. Thessalonikerbrev",
    "1tim": "1. Timotheusbrev",
    "2tim": "2. Timotheusbrev",
    "tit": "Titusbrevet",
    "filem": "Filemonbrevet",
    "hebr": "Hebræerbrevet",
    "jak": "Jakobsbrevet",
    "1pet": "1. Petersbrev",
    "2pet": "2. Petersbrev",
    "1joh": "1. Johannesbrev",
    "2joh": "2. Johannesbrev",
    "3joh": "3. Johannesbrev",
    "jud": "Judas' Brev",
    "aab": "Johannes' Åbenbaring",
}


@dataclass(frozen=True)
class Verse:
    """A single verse. `index` is its position in the corpus, used everywhere as its id."""
    index: int
    book: str
    chapter: int
    verse: int
    text: str

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.book, self.chapter, self.verse)


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split())).strip()


class Corpus:
    """The verse corpus plus the lookups every retrieval script needs.

    >>> corpus = Corpus.load()
    >>> len(corpus)
    31170
    >>> corpus.reference(corpus.index_of("es", 66, 24))
    "Esajas' Bog 66:24"
    """

    def __init__(self, verses: list[Verse], book_names: dict[str, str]):
        self.verses = verses
        self.book_names = book_names
        self.documents = [v.text for v in verses]
        self._by_key = {v.key: v.index for v in verses}
        # The Bible sometimes repeats itself verbatim (Højsangen 2:6 == 8:3, Markus 9:44 == 9:46).
        self._by_text: dict[str, set[int]] = collections.defaultdict(set)
        for v in verses:
            self._by_text[_normalize_text(v.text)].add(v.index)


    @classmethod
    def load(cls) -> "Corpus":
        """Read the corpus from `data/all_verses_ot_nt.json`."""
        raw = json.loads(data_file(VERSES).read_text(encoding="utf-8"))
        verses = [Verse(i, v["book"], v["chapter"], v["verse"], v["text"])
                  for i, v in enumerate(raw)]
        return cls(verses, dict(BOOK_NAMES))

    def __len__(self) -> int:
        return len(self.verses)

    def __getitem__(self, index: int) -> Verse:
        return self.verses[index]

    def __iter__(self) -> Iterator[Verse]:
        return iter(self.verses)


    def index_of(self, book: str, chapter: int, verse: int) -> int | None:
        return self._by_key.get((book, chapter, verse))

    def span(self, book: str, chapter: int, first: int, last: int | None = None) -> set[int]:
        last = first if last is None else last
        found = (self.index_of(book, chapter, v) for v in range(first, last + 1))
        return {i for i in found if i is not None}

    def duplicates_of(self, indices: set[int]) -> set[int]:
        result: set[int] = set()
        for i in indices:
            result |= self._by_text[_normalize_text(self.verses[i].text)]
        return result

    def reference(self, index: int) -> str:
        v = self.verses[index]
        return f"{self.book_names.get(v.book, v.book)} {v.chapter}:{v.verse}"
