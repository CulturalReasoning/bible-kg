# Retrieving Biblical Intertextual References in Karen Blixen’s Seven Gothic Tales

Given a passage from Karen Blixen's *Syv fantastiske Fortællinger* (1935), retrieve
the Bible verse it refers to, from the full Danish Bible - 31,170 verses of
GT1871 / NT1907.

This repository is an artifact of most of the code used as the retrieval and evaluation
core of the research. As the studied work is still protected by copyright, the gold references could not be shared. 

---


## Results

Metrics are P@1, R@10, MRR@10, NDCG@10, all at k=10, over the 189 references.
R@10 — *did a gold verse make the top 10* - is the one examined most often

### Lexical retrieval

| retriever | P@1 | R@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| **bm25 + lemma&nbsp;+&nbsp;stem** | **0.180** | **0.365** | **0.240** | **0.270** |
| tfidf + lemma + stem | 0.138 | 0.302 | 0.193 | 0.219 |
| bm25 + raw tokens | 0.164 | 0.286 | 0.208 | 0.227 |
| tfidf + raw tokens | 0.175 | 0.286 | 0.214 | 0.231 |

### Dense retrieval, zero-shot

| model | P@1 | R@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| multilingual-e5-large | 0.138 | 0.360 | 0.207 | 0.243 |
| multilingual-e5-base | 0.116 | 0.286 | 0.158 | 0.187 |
| dfm-sentence-encoder-large | 0.116 | 0.265 | 0.158 | 0.183 |
| multilingual-e5-small | 0.095 | 0.206 | 0.129 | 0.148 |
| dfm-sentence-encoder-small | 0.037 | 0.063 | 0.043 | 0.048 |

### Dense retrieval, fine-tuned

`dfm-sentence-encoder-large`, contrastively fine-tuned with hard negatives, 5-fold
grouped cross-validation

| metric | zero-shot | fine-tuned | Δ |
|---|---:|---:|---:|
| P@1 | 0.115 | **0.338** ± 0.035 | +0.223 |
| R@10 | 0.274 | **0.507** ± 0.066 | +0.233 |
| MRR@10 | 0.159 | **0.390** ± 0.043 | +0.231 |
| NDCG@10 | 0.186 | **0.418** ± 0.048 | +0.232 |

---