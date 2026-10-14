# Code Retrieval research specification

> Active scope: a CosQA-only reproduction and ablation study. The versioned controls are in [study-protocol-2026-10-14.md](study-protocol-2026-10-14.md). Historical results remain preserved in [reproducibility.md](reproducibility.md) and the provenance audit.

## Question

On the fixed CoIR CosQA snapshot, how do the existing E5, HyDE, rank-fusion, and reranking conditions affect `nDCG@10`, and can the saved comparison be reproduced under its recorded controls?

## Scope and claims

Use one dataset: CoIR-Retrieval/cosqa. This study explains the existing CosQA results and isolates the contribution of the two HyDE representations, fusion, and reranker choice.

The validation and test metrics have already been examined. New results on the test split are retrospective and exploratory, not independent confirmatory evidence. This protocol does not support cross-dataset generalization, broad code-search claims, novelty, or state-of-the-art claims.

R1 explicitly offered a focused CosQA study as an alternative to a second dataset; that is the selected path. R1's code-oriented model comparison is planned through a code-trained reranker. R3's second-dataset request and concern about Flan-T5-base hypothesis quality remain unresolved and must be reported as limitations; this protocol does not add a second dataset or benchmark a stronger generator.

## Dataset and evaluator

- Dataset: `CoIR-Retrieval/cosqa`, immutable revision `0846fa3b963a21bead36e9fab61451fc83b777a6`.
- Use the `queries/queries` and `corpus/corpus` populations with the `default` qrels configuration. Use the complete `valid` and `test` qrels splits: 500 queries each and the complete 20,604-document corpus.
- Preserve source IDs. Use text-only query and corpus inputs; exclude titles and other metadata.
- Use the existing E5 setup: `intfloat/e5-base-v2` at revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, query/passage prefixes, normalized embeddings, 512-token limit, exact inner-product retrieval, candidate depth 1,000, seed 42, batch size 128, and 8 CPU threads.
- Evaluate with `coir-eval==0.7.0` and the existing saved-order tie policy. Record that the local policy differs from an unqualified stock CoIR leaderboard result.

## Comparison matrix

Keep the original four rows as the reproduction baseline. All rows use the same data population, evaluator, candidate depth, and E5 first stage unless the representation is the variable under study.

| Condition | Query representation or reranker | Purpose |
| --- | --- | --- |
| `e5` | Original query | Baseline |
| `e5_hyde` | Original query plus generated hypothesis; weighted RRF with the original E5 ranking | Existing fused HyDE condition |
| `e5_rerank` | Existing general-purpose cross-encoder on the E5 top-1,000; weighted RRF with the E5 order | Existing reranker condition |
| `e5_hyde_rerank` | Weighted RRF of the selected HyDE and generic-reranker rankings | Existing combined condition |
| `e5_hyde_raw` | Original query plus generated hypothesis, with no rank fusion | Representation/fusion ablation; reuse the existing raw ranking if its provenance matches |
| `e5_hypothesis_only` | Generated hypothesis alone (canonical HyDE), with no rank fusion | Representation ablation using the same saved hypotheses |
| `e5_code_rerank` | Code-trained pair scorer on the same E5 top-1,000, with a validation-selected E5 fusion | Code-oriented reranker comparison |

Report each reranker's raw pair-score order as a diagnostic. Preserve candidate IDs and E5 order for score ties; rerankers must not add documents. Do not expand this into a full generator-by-reranker factorial.

## Generation and fusion controls

Use the existing `google/flan-t5-base` generator at revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`. Generate one hypothesis from query text only, using the current prompt and fallback sequence in `code-retrieval/src/e5_hyde_rerank.py`, greedy decoding (`do_sample=false`), EOS stopping, and a 64-token limit. Preserve the exact generated text, fallback use, and cache identity. Empty output is a failed condition, not grounds for a silent substitute.

For query-plus-hypothesis input, preserve the original implementation's trimmed query, two LF characters, and trimmed hypothesis. For the canonical HyDE ablation, use only that same trimmed hypothesis. Do not regenerate hypotheses between these representation conditions.

Use the existing validation-only weighted-RRF selection: all 500 `valid` queries, `rrf_k=60`, and weights from 0.00 through 1.00 in 0.05 steps. On a tie, prefer the larger weight on the first input. Record every tested weight and the selected setting. Do not use test results to pick prompts, weights, models, or conditions.

## Code-trained reranker and compute

Compare `NamanAgnih0tri/code-reranker-miniLM-staqc` at pinned revision `9bcf5f1cceef4914e2fc564b2e4ee78410626702`. Its model card reports code-pair training; treat that statement as publisher-reported, not independently audited. Before a complete comparison, load the real weights on CPU and verify one finite score per query/code pair. If loading or scoring fails, or available local resources are insufficient, stop and document the limitation. Do not switch to a remote GPU, quantize, sample the benchmark, or silently substitute a different checkpoint.

All inference in this study is local CPU inference. Record the CPU, RAM, package versions, batch/thread settings, per-stage runtime, and peak host memory. GPU memory is not applicable. A smoke or preflight run is not benchmark evidence.

## Outcomes and interpretation

- Primary metric: mean per-query `nDCG@10`, with absolute scores and paired deltas against E5.
- Secondary metrics: `MAP@10`, `Recall@10`, and `Recall@1000`, using the existing metric implementation.
- Report the existing paired query-bootstrap summaries using 20,000 resamples, seed 518, and unadjusted 95% percentile intervals. Treat them as descriptive, not confirmatory or multiplicity-adjusted inference.
- Preserve per-query rows, complete ordered rankings, generation records, and provenance for every condition.
- Negative, null, and failed outcomes are valid. Do not infer expansion effectiveness from a fused score alone; report raw and fused conditions separately.

## Historical evidence and stopping rules

The saved CosQA test metrics and validation choices have been reviewed. Keep them labeled as historical. New runs use unique output identities and may not overwrite prior artifacts. The exact conference-submission experiment provenance is still unresolved.

No second dataset is part of this study. The request for one remains an acknowledged reviewer limitation; do not imply that the CosQA-only study satisfies it. Do not use APPS, Qwen, remote-GPU resources, or new human labels in this protocol.
