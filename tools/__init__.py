"""
toolkit for the following intertextuality retrieval task
given a passage from *Syv fantastiske Fortællinger*, find the verse it refers to in the
31,170-verse Danish Bible corpus (GT1871 + NT1907), evaluated against 189 gold
references from the scholarly commentary.

example use:

    from tools import Corpus, GoldSet
    from tools.sparse import BM25Retriever, rank_all
    from tools.text import load_dacy_tokens
    from tools.metrics import by_band

    corpus = Corpus.load()
    gold = GoldSet.load(corpus)
    tokens = load_dacy_tokens()

    bm25 = BM25Retriever(tokens.corpus_stem)
    ranks = rank_all(bm25, [tokens.query_stem[r.id] for r in gold], gold)
    print(by_band(gold, ranks))

Modules
paths     data config
corpus    bible corpus lookup
gold      the 189 gold references from the scholarly edition
text      normalisation for 19th-century Danish 
metrics   P@1 / R@10 / MRR@10 / NDCG@10 metrics and calculations
sparse    TF-IDF and BM25 methods
dense     sentence-transformers retrieval methods
finetune  fine-tuning with cv
bws       building the best-worst-scaling tuples for annotation (expert/LLM)
"""

from .corpus import Corpus, Verse
from .gold import BANDS, GoldSet, Reference

__all__ = ["Corpus", "Verse", "GoldSet", "Reference", "BANDS"]
__version__ = "1.0.0"