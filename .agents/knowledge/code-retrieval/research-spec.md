# Code Retrieval research specification

**Status:** Scope selected on 2026-10-14. No new model inference or benchmark run has been performed under this specification.

This file defines the study scope and its exact execution procedure; use it as the canonical specification.

## Question, scope, and claims

On the fixed CoIR CosQA snapshot, how do the existing E5, HyDE, rank-fusion, and reranking conditions affect `nDCG@10`, and can the saved comparison be reproduced under its recorded controls?

This is a single-dataset reproducibility and ablation study. It explains the existing CosQA results and separates the effects of query representation, rank fusion, and reranker choice. It does not establish broad code-search generalization, novelty, or state of the art.

CosQA test metrics and validation-selected weights have already been examined. New test-split results are retrospective and exploratory, not independent confirmatory evidence. The exact experiment used in the conference submission remains unresolved.

**Reviewer boundary:** R1 explicitly allowed either another dataset or a narrowly scoped CosQA reproducibility/ablation study; this specification selects the CosQA-only path. R1's code-oriented model comparison is planned through a code-trained reranker. R3's requested representation ablations are included. R3's second-dataset request and concern about Flan-T5-base hypothesis quality remain unresolved and are outside scope. This study uses CosQA only; APPS and Qwen runs are also out of scope. The existing pinned Flan-T5-base is retained to reproduce the CosQA workflow, not to establish generator quality or cross-dataset generalization. Whether this narrowed scope satisfies the reviewers remains for them to decide.

## Dataset and evaluation boundary

- Dataset: [CoIR-Retrieval/cosqa](https://huggingface.co/datasets/CoIR-Retrieval/cosqa/tree/0846fa3b963a21bead36e9fab61451fc83b777a6), immutable revision `0846fa3b963a21bead36e9fab61451fc83b777a6`.
- Use `queries/queries`, `corpus/corpus`, and the `default` qrels configuration. Evaluate the complete `valid` and `test` splits: 500 queries each, over the complete 20,604-document corpus.
- Preserve source IDs and use text-only query and corpus fields. Exclude titles and other metadata from model inputs.
- Use only the pinned qrels; do not add or solicit human relevance labels.
- The pinned valid and test query IDs are disjoint. This does not establish which exact selection and test artifacts were used in the conference submission.
- Use `valid` for selection and `test` for descriptive evaluation only. Never use test results to choose prompts, models, weights, or conditions. Because test results have already been reviewed, label all new test comparisons retrospective/exploratory.
- Use the existing pinned `coir-eval==0.7.0` evaluation path. Preserve the repository's saved insertion order for ties and its unique-ordinal-score conversion. State this local policy; do not call it an unqualified reproduction of a stock CoIR leaderboard score.

## Pinned models and controls

| Component | Checkpoint and revision | Frozen controls |
| --- | --- | --- |
| First-stage retriever | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2/tree/f52bf8ec8c7124536f0efb74aca902b2995e5bcd), `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | Existing `query: ` / `passage: ` prefixes, 512-token limit, normalized embeddings, exact inner-product retrieval, depth 1,000, batch size 128, seed 42, 8 CPU threads. |
| HyDE generator | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base/tree/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2), `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | One hypothesis from query text only; existing prompt and fallbacks; tokenizer input truncation at 512 tokens; greedy decoding (`do_sample=false`; configured temperature `1.0` is inactive); EOS stopping; maximum 64 new tokens; empty output is an error. |
| Existing general reranker | [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a), `233902d25c440f23af6f7d6e94d2946bac0bee0a` | Score the E5 top-1,000 candidates; max input length 512; CPU inference; preserve the candidate IDs and E5 order for score ties. |
| Code-trained comparator | [`NamanAgnih0tri/code-reranker-miniLM-staqc`](https://huggingface.co/NamanAgnih0tri/code-reranker-miniLM-staqc/tree/9bcf5f1cceef4914e2fc564b2e4ee78410626702), `9bcf5f1cceef4914e2fc564b2e4ee78410626702` | Planned alternative reranker on the same E5 top-1,000 candidates, CPU only, after the real-weight preflight below. |

The code-reranker model card describes StaQC question-to-code training and a pair-scoring configuration. Treat that description as publisher-reported, not independently audited, and do not use its reported scores as evidence. Before a complete run, load the pinned real weights on CPU and verify one finite score per query/code pair. If load or scoring fails, or local memory/time is insufficient, stop and document the limitation. Do not change the checkpoint, quantize, reduce the dataset, or move to remote-GPU execution.

The existing prompt and fallback sequence live in `code-retrieval/src/e5_hyde_rerank.py`. The main prompt is “Answer this programming question with a concise solution: {query}”. The fallbacks are “Provide a short answer to the following: {query}”, “Python solution: {query}”, “Write code for: {query}”, and “Give a Python example for: {query}”. Store the exact generated text, fallback use, token counts, and cache identity. The generator must not receive qrels, labels, answers, or corpus documents.

The existing query-plus-hypothesis representation trims the original query, joins it to the trimmed hypothesis with two LF characters, and reuses the saved hypothesis. The canonical HyDE ablation uses the trimmed hypothesis alone. Do not regenerate hypotheses between these representation conditions.

## System matrix and ablations

Keep the original four rows as the reproduction baseline:

| ID | Condition |
| --- | --- |
| `e5` | Original query with the fixed E5 first stage. |
| `e5_hyde` | Retrieve with original query plus generated hypothesis, then weighted-RRF fuse with the original E5 ranking. |
| `e5_rerank` | Score and reorder only the original E5 top-1,000 with the existing general reranker, then weighted-RRF fuse with E5. |
| `e5_hyde_rerank` | Weighted RRF of the selected `e5_hyde` and `e5_rerank` rankings. |

Add only these focused conditions:

| ID | Condition |
| --- | --- |
| `e5_hyde_raw` | Original query plus the same generated hypothesis, without rank fusion. Reuse the existing raw ranking if its provenance matches; otherwise preserve a new run in the existing notebook workflow. |
| `e5_hypothesis_only` | E5 retrieval using only the same generated hypothesis (canonical HyDE), with no rank fusion. |
| `e5_code_rerank` | Replace the existing general reranker with the pinned code-trained pair scorer on the same E5 candidate IDs. Report its raw order and its validation-selected fusion with E5. |

Do not add a stronger generator or a full generator-by-reranker factorial. Keep all new conditions in the existing notebook and reusable modules; do not create a parallel notebook or one-off script.

For both rerankers, report raw pair-score order as a diagnostic, preserve the E5 candidate IDs and E5 order for score ties, and never add documents outside the original E5 top-1,000 pool.

Use weighted reciprocal-rank fusion with `rrf_k=60`. For each fusion setting, sweep the first-input weight over 0.00 to 1.00 in 0.05 increments on all 500 validation queries only. Prefer the larger first-input weight on ties. Preserve every tested weight, selected weight, and selection artifact digest. The currently saved historical selections remain attached to their original run identities; do not relabel them.

## Outcomes and statistics

- Primary outcome: mean per-query `nDCG@10`; report each condition's absolute score and paired delta versus E5.
- Secondary outcomes: `MAP@10`, `Recall@10`, and `Recall@1000`, from the same ordered rankings and qrels.
- Use the existing paired query-bootstrap summary: 20,000 resamples, seed 518, 95% percentile intervals. These intervals are descriptive and unadjusted; do not present them as confirmatory or simultaneous bounds.
- Preserve complete per-query metric rows, ranking order, candidate IDs, exclusions, generation records, selection outputs, and run metadata. Report runtime and peak host memory for the CPU stages; record unavailable measurements rather than inventing them.
Do not infer that query expansion helped from a fused score alone; report raw and fused HyDE conditions separately.

- Negative, null, failed, and partial results are valid outcomes. Smoke and preflight runs are not benchmark evidence.

## Compute and stopping rules

- Use local CPU inference only. Existing E5, Flan-T5-base, and general-reranker workflows already specify CPU execution. The code-trained model must pass its local CPU preflight before a complete comparison.
- Use the existing batch and thread controls where they apply. Record exact CPU/RAM, Python and package versions, model revisions, device, batch size, thread count, elapsed time per stage, and peak host memory.
- If a pinned model cannot load or available local resources cannot support the complete declared population, stop before the complete run and report the limitation. Do not shrink the query set or corpus, lower candidate depth, substitute weights, quantize, or use paid/remote GPU compute.
- A complete run uses all 500 selected-split queries when selection is required and all 500 test queries with the complete 20,604-document corpus. Partial or failed executions remain partial or failed and cannot support a quality claim.

## Historical evidence and source record

Preserve earlier CosQA results, code revisions, selection artifacts, caches, and run identities. Give every new run a unique output identity; never overwrite prior artifacts. The test outcomes already reviewed remain retrospective; this specification cannot turn them into an untouched holdout. The saved experiment provenance for the conference submission is unresolved.

- CoIR paper: https://aclanthology.org/2025.acl-long.1072.pdf
- CosQA dataset at the pinned revision: https://huggingface.co/datasets/CoIR-Retrieval/cosqa/tree/0846fa3b963a21bead36e9fab61451fc83b777a6
- Existing experiment entry point: `code-retrieval/notebooks/e5_hyde_rerank_experiment.ipynb`
- Existing generation and fusion implementation: `code-retrieval/src/e5_hyde_rerank.py`
- Existing CosQA provenance report: `experiment-provenance-audit.md`
