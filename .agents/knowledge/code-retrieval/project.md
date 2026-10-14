# Code Retrieval research project

## Objective

Evaluate how code-oriented HyDE query expansion and pairwise code reranking affect
`intfloat/e5-base-v2` on APPS text-to-code contest retrieval. The completed CosQA
study remains retrospective descriptive evidence.

## Study contract

| Item | Value |
| --- | --- |
| Repository | `https://github.com/azhry/research.git` |
| Directory | `code-retrieval` |
| Language/runtime | Python; Google Colab / Jupyter Notebook |
| Benchmark | `https://github.com/coir-team/coir` |
| Prospective dataset | `CoIR-Retrieval/apps` at `ad8755dfa53f1de865e165d507fa0523502b5682` |
| Historical dataset | `CoIR-Retrieval/cosqa` at `0846fa3b963a21bead36e9fab61451fc83b777a6` |
| Retriever | `intfloat/e5-base-v2` |
| HyDE generator | `Qwen/Qwen2.5-Coder-7B-Instruct` at `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| Pairwise reranker | `NamanAgnih0tri/code-reranker-miniLM-staqc` at `9bcf5f1cceef4914e2fc564b2e4ee78410626702` |
| Primary metric | `nDCG@10` |
| Linear project | `Code Retrieval` |

The complete frozen scope, selection split, prompts, system matrix, evaluation
boundary, multiplicity policy, resource cap, and data-handling constraints are in
[`study-protocol-2026-10-14.md`](study-protocol-2026-10-14.md). This protocol is
approved but not yet implemented or benchmarked.

## Required experiments

1. E5 baseline
2. E5 + HyDE
3. E5 + re-ranking
4. E5 + HyDE + re-ranking

All systems use the same corpus, split, preprocessing, candidate depth, relevance
judgments, and evaluator. HyDE changes only the query representation; the
cross-encoder only reorders the first-stage candidate pool.

The APPS generator, prompt, pair-scoring reranker, and settings are fixed by the
revised protocol. Do not change them based on test results.

## Deliverables

- Top-to-bottom executable notebook.
- Reusable loading, retrieval, HyDE, re-ranking, and evaluation code.
- One machine-readable result table covering all four systems.
- Run metadata: commit/notebook hash, versions, hardware, seed, revisions, and
  parameters.
- Interpretation of measured `nDCG@10` changes, cost, and limitations.

Do not mix the legacy CodeBERT training notes into this comparison unless a separate
experiment explicitly defines that baseline.
