"""Fine-tuning a Danish sentence encoder on the 189 gold references.

The training data is a tuple `(Blixen passage, gold verse, hard negative)`

hard negatives from both sparse and dense methods, but never a different verse from the same commentary entry
"""
from __future__ import annotations

import gc
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .corpus import Corpus
from .dense import DenseRetriever
from .gold import BANDS, Reference
from .metrics import METRICS, metrics_from_ranks
from .paths import ARTIFACTS, ensure_dir
from .sparse import BM25Retriever
from .text import prepare_document, prepare_query, tokenize


@dataclass
class FinetuneConfig:
    model: str = "KennethEnevoldsen/dfm-sentence-encoder-large"
    use_ortho: bool = True
    clean_queries: bool = True

    # unsupervised domain adaptation: "tsdae" | "simcse" | "none"
    pretrain: str = "none"
    pretrain_epochs: int = 1
    tsdae_batch: int = 16
    simcse_batch: int = 64

    # supervised step
    folds: int = 5
    seed: int = 42
    epochs: int = 2
    batch_size: int = 64
    mini_batch_size: int = 32       # for CachedMultipleNegativesRankingLoss
    learning_rate: float = 2e-5
    encode_batch_size: int = 256

    # negative mining
    n_hard_negatives: int = 16
    max_positives: int = 3          
    bm25_band: tuple[int, int] = (3, 50)
    dense_pool: int = 60

    n_seeds: int = 3

    device: str | None = None

    def describe(self) -> str:
        return (f"{self.model.split('/')[-1]} | ortho={self.use_ortho} "
                f"clean_q={self.clean_queries} | pretrain={self.pretrain} | "
                f"{self.folds}-fold | soup x{self.n_seeds}")


class NegativeMiner:
    def __init__(self, corpus: Corpus, config: FinetuneConfig, documents: list[str]):
        self.config = config
        self.corpus = corpus
        self._bm25 = BM25Retriever([tokenize(d) for d in documents])
        self._bm25_cache: dict[str, list[int]] = {}

    def _bm25_candidates(self, reference: Reference) -> list[int]:
        if reference.id not in self._bm25_cache:
            start, stop = self.config.bm25_band
            self._bm25_cache[reference.id] = self._bm25.band(
                tokenize(reference.text), start, stop)
        return self._bm25_cache[reference.id]

    def mine(self, references: list[Reference],
             dense_pool: dict[str, list[int]]) -> dict[str, list[int]]:
        negatives: dict[str, list[int]] = {}
        for reference in references:
            excluded = self.corpus.duplicates_of(reference.sibling_gold)
            picked: list[int] = []
            for candidate in list(dense_pool[reference.id]) + self._bm25_candidates(reference):
                candidate = int(candidate)
                if candidate not in excluded and candidate not in picked:
                    picked.append(candidate)
                if len(picked) >= self.config.n_hard_negatives:
                    break
            negatives[reference.id] = picked
        return negatives


def build_examples(references: list[Reference], negatives: dict[str, list[int]],
                   documents: list[str], config: FinetuneConfig):
    from sentence_transformers import InputExample

    examples = []
    for reference in references:
        query = prepare_query(reference.text, config.use_ortho, config.clean_queries)
        for positive in sorted(reference.gold)[: config.max_positives]:
            for negative in negatives[reference.id]:
                examples.append(InputExample(
                    texts=[query, documents[positive], documents[negative]]))
    return examples


def domain_pretrain(documents: list[str], config: FinetuneConfig) -> str:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    if config.pretrain == "none":
        return config.model

    def run(kind: str):
        model = SentenceTransformer(config.model, device=config.device)
        if kind == "tsdae":
            from sentence_transformers.datasets import DenoisingAutoEncoderDataset
            from sentence_transformers.losses import DenoisingAutoEncoderLoss

            loader = DataLoader(DenoisingAutoEncoderDataset(list(documents)),
                                batch_size=config.tsdae_batch, shuffle=True)
            loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path=config.model,
                                            tie_encoder_decoder=True)
            model.fit([(loader, loss)], epochs=config.pretrain_epochs, weight_decay=0,
                      scheduler="constantlr", optimizer_params={"lr": 3e-5},
                      show_progress_bar=False)
        else:
            loader = DataLoader([InputExample(texts=[s, s]) for s in documents],
                                batch_size=config.simcse_batch, shuffle=True)
            model.fit([(loader, losses.MultipleNegativesRankingLoss(model))],
                      epochs=config.pretrain_epochs, optimizer_params={"lr": 3e-5},
                      show_progress_bar=False)
        return model

    try:
        model, used = run(config.pretrain), config.pretrain
    except Exception as exc:                   
        print(f"  {config.pretrain} failed, falling back to SimCSE: {str(exc)[:120]}")
        model, used = run("simcse"), "simcse"

    path = str(ensure_dir(ARTIFACTS) / "domain_pretrained")
    model.save(path)
    del model
    _free_memory(config.device)
    print(f"  domain pretraining ({used}, {len(documents)} sentences) to {path}")
    return path


def train_one(base_checkpoint: str, references: list[Reference], documents: list[str],
              miner: NegativeMiner, dense_pool: dict[str, list[int]],
              config: FinetuneConfig, seed: int):
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = SentenceTransformer(base_checkpoint, device=config.device)
    try:
        loss = losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=config.mini_batch_size)
    except Exception:                             # older sentence-transformers
        loss = losses.MultipleNegativesRankingLoss(model)

    examples = build_examples(references, miner.mine(references, dense_pool),
                              documents, config)
    loader = DataLoader(examples, shuffle=True, batch_size=config.batch_size)
    model.fit(train_objectives=[(loader, loss)], epochs=config.epochs,
              optimizer_params={"lr": config.learning_rate},
              warmup_steps=int(0.1 * len(loader) * config.epochs),
              use_amp=(config.device == "cuda"), show_progress_bar=False)
    return model


def train_soup(base_checkpoint: str, references: list[Reference], documents: list[str],
               miner: NegativeMiner, dense_pool: dict[str, list[int]],
               config: FinetuneConfig):
    from sentence_transformers import SentenceTransformer

    if config.n_seeds <= 1:
        return train_one(base_checkpoint, references, documents, miner,
                         dense_pool, config, config.seed)

    accumulated = None
    for offset in range(config.n_seeds):
        model = train_one(base_checkpoint, references, documents, miner,
                          dense_pool, config, config.seed + offset)
        state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
        accumulated = state if accumulated is None else {
            k: accumulated[k] + state[k] for k in accumulated}
        del model
        _free_memory(config.device)

    soup = SentenceTransformer(base_checkpoint, device=config.device)
    target = soup.state_dict()
    soup.load_state_dict({k: (accumulated[k] / config.n_seeds).to(dtype=target[k].dtype)
                          for k in target})
    return soup


def cross_validate(corpus: Corpus, gold, config: FinetuneConfig,
                   checkpoint_path: Path | None = None) -> dict:
    import json

    import torch

    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    documents = [prepare_document(d, config.use_ortho) for d in corpus.documents]
    references = list(gold)

    # for the hard negatives
    base = DenseRetriever(config.model, use_ortho=config.use_ortho,
                          clean_queries=config.clean_queries, device=config.device,
                          batch_size=config.encode_batch_size)
    base.encode_corpus(corpus.documents)
    zero_shot_ranks, zero_shot_top = base.ranks_and_top(references, k=config.dense_pool)
    zero_shot = {r.id: rank for r, rank in zip(references, zero_shot_ranks)}
    dense_pool = {r.id: top for r, top in zip(references, zero_shot_top)}
    print(f"  {metrics_from_ranks(zero_shot_ranks)}")
    del base
    _free_memory(config.device)

    miner = NegativeMiner(corpus, config, documents)
    base_checkpoint = domain_pretrain(documents, config)

    folds, out_of_fold, out_of_fold_top = [], {}, {}
    for fold, (train_refs, test_refs) in enumerate(
            gold.stratified_group_folds(config.folds, config.seed)):
        model = train_soup(base_checkpoint, train_refs, documents, miner, dense_pool, config)

        retriever = DenseRetriever.from_model(
            model, config.model, use_ortho=config.use_ortho,
            clean_queries=config.clean_queries, batch_size=config.encode_batch_size)
        retriever.corpus_embeddings = retriever.encode(
            [retriever.doc_prefix + d for d in documents])
        ranks, tops = retriever.ranks_and_top(test_refs, k=100)

        for reference, rank, top in zip(test_refs, ranks, tops):
            out_of_fold[reference.id] = rank
            out_of_fold_top[reference.id] = top
        folds.append({
            "fold": fold,
            "n_test": len(test_refs),
            "zero_shot": _by_band(test_refs, [zero_shot[r.id] for r in test_refs]),
            "finetuned": _by_band(test_refs, ranks),
        })
        before = folds[-1]["zero_shot"]["overall"]["R@10"]
        after = folds[-1]["finetuned"]["overall"]["R@10"]
        print(f"fold {fold}: R@10 {before:.3f} -> {after:.3f} ({after - before:+.3f})")

        if checkpoint_path:
            checkpoint_path.write_text(json.dumps(
                {"folds": folds, "out_of_fold": out_of_fold}, ensure_ascii=False))
        del model, retriever
        _free_memory(config.device)

    return {
        "config": asdict(config),
        "n_references": len(references),
        "folds": folds,
        "summary": _summarise(folds),
        "zero_shot_ranks": zero_shot,
        "out_of_fold_ranks": out_of_fold,
        "out_of_fold_top": out_of_fold_top,
    }


def _by_band(references: list[Reference], ranks: list[int]) -> dict:
    """Metrics for a fold, overall and per relation band."""
    result = {"overall": {**metrics_from_ranks(ranks), "n": len(ranks)}}
    for band in BANDS:
        selected = [ranks[i] for i, r in enumerate(references) if r.band == band]
        result[band] = {**metrics_from_ranks(selected), "n": len(selected)}
    return result


def _summarise(folds: list[dict]) -> dict:
    """Mean (and fine-tuned std) of each metric across folds."""
    summary = {}
    for metric in METRICS:
        before = [f["zero_shot"]["overall"][metric] for f in folds]
        after = [f["finetuned"]["overall"][metric] for f in folds]
        summary[metric] = {"zero_shot": float(np.nanmean(before)),
                           "finetuned": float(np.nanmean(after)),
                           "finetuned_std": float(np.nanstd(after))}
    return summary


def _free_memory(device: str | None) -> None:
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()