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
