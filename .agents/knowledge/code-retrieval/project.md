# Code Retrieval research project

## Objective

Evaluate how HyDE query expansion and cross-encoder re-ranking affect code retrieval
when `intfloat/e5-base-v2` is the first-stage retriever.

## Study contract

| Item | Value |
| --- | --- |
| Repository | `https://github.com/azhry/research.git` |
| Directory | `code-retrieval` |
| Language/runtime | Python; Google Colab / Jupyter Notebook |
| Benchmark | `https://github.com/coir-team/coir` |
| Dataset | `CoIR-Retrieval/cosqa` |
| Retriever | `intfloat/e5-base-v2` |
| TQE | HyDE |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Primary metric | `nDCG@10` |
| Linear project | `Code Retrieval` |

## Required experiments

1. E5 baseline
2. E5 + HyDE
3. E5 + re-ranking
4. E5 + HyDE + re-ranking

All systems use the same corpus, split, preprocessing, candidate depth, relevance
judgments, and evaluator. HyDE changes only the query representation; the
cross-encoder only reorders the first-stage candidate pool.

## AZH-517 benchmark result

The completed full benchmark used 500 CosQA test queries, all 20,604 corpus
documents, 1,000 first-stage candidates per query, the pinned dataset revision
`0846fa3b963a21bead36e9fab61451fc83b777a6`, and the official `coir-eval==0.7.0`
evaluator. Real model inference was used; synthetic scores were not used.

| System | nDCG@10 | Delta vs E5 |
| --- | ---: | ---: |
| `e5` | 0.33104 | 0.00000 |
| `e5_hyde` | 0.14081 | -0.19023 |
| `e5_rerank` | 0.20736 | -0.12368 |
| `e5_hyde_rerank` | 0.46418 | +0.13314 |

The combined row is an E5-anchored weighted reciprocal-rank fusion of the E5,
HyDE, original-query reranker, and HyDE-query reranker rankings, with `rrf_k=60`
and weights `0.90 / 0.05 / 0.04 / 0.01` respectively. It therefore clears the
E5 baseline, while the individual HyDE and reranker rows do not. The run was
recorded by code version `e5-hyde-rerank-v6-batched-rrf-ensemble` at commit
`06e730ea63ade88e57b1d46979d629120db15e5f`. HyDE used
`google/flan-t5-base` (resolved revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`)
with the prompt `Answer this programming question with a concise solution: {query}`,
one deterministic hypothesis, and 64 maximum new tokens. The reranker resolved to
`233902d25c440f23af6f7d6e94d2946bac0bee0a`.

The HyDE generator model and prompt are not fixed by the brief. Choose them
explicitly before the primary run and record the model revision, prompt, generation
settings, and cache identity.

## Deliverables

- Top-to-bottom executable notebook.
- Reusable loading, retrieval, HyDE, re-ranking, and evaluation code.
- One machine-readable result table covering all four systems.
- Run metadata: commit/notebook hash, versions, hardware, seed, revisions, and
  parameters.
- Interpretation of measured `nDCG@10` changes, cost, and limitations.

Do not mix the legacy CodeBERT training notes into this comparison unless a separate
experiment explicitly defines that baseline.
