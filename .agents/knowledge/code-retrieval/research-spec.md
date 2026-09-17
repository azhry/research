# Code Retrieval research specification

## Question

On CosQA, how much do HyDE and a cross-encoder re-ranker improve E5 code retrieval,
individually and together, according to `nDCG@10`?

## System matrix

| ID | First stage | TQE | Re-ranker |
| --- | --- | --- | --- |
| `e5` | `intfloat/e5-base-v2` | None | None |
| `e5_hyde` | `intfloat/e5-base-v2` | HyDE | None |
| `e5_rerank` | `intfloat/e5-base-v2` | None | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `e5_hyde_rerank` | `intfloat/e5-base-v2` | HyDE | `cross-encoder/ms-marco-MiniLM-L6-v2` |

The re-ranker receives only the shared first-stage candidate pool and may reorder it;
it must not introduce new documents.

## Dataset and evaluator

- Dataset: `CoIR-Retrieval/cosqa`.
- Benchmark: COIR repository.
- Record exact dataset split, revision, loaded field names, benchmark commit/package
  version, and relevance-judgment convention.
- Inspect the real schema before writing adapters; do not rely on legacy local notes.

## HyDE controls

Expose and record the generator model/revision, prompt, number of hypotheses,
temperature, maximum tokens, stopping behavior, combination strategy, and cache key.
The original query remains the evaluation query. HyDE must not access labels,
answers, or target documents.

## Primary result

Report aggregated `nDCG@10` for all four systems, absolute and delta versus E5,
query count/exclusions, candidate and re-ranking depth, and whether the official
COIR evaluator completed successfully. Latency and memory are secondary observations.

Keep split, corpus snapshot, preprocessing, model revisions, prefixes, truncation,
candidate depth, evaluator, and seed fixed unless an exploratory run explicitly
documents a deviation.
