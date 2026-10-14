# AZH-526 revised code retrieval protocol

**Status:** Frozen on 2026-10-14 after the user selected the recommended defaults. This is a design record; no APPS inference or benchmark run has been performed under it.

## Question, scope, and claims

For text-to-code contest retrieval on APPS, do a code-oriented HyDE query and a code-trained pair-scoring reranker, separately and together, change E5-base-v2 `nDCG@10`?

This is the expanded study. APPS is prospective relative to this study's decisions: its test scores have not been used to select this protocol. Existing CosQA results remain retrospective descriptive evidence because its test scores were already examined and reported. Do not pool the two datasets or use CosQA test results to select APPS settings. Model pretraining contamination on APPS is unknown, so the test is not claimed to be unseen by the checkpoints. The claim boundary is APPS text-to-code contest retrieval plus the separately labeled historical CosQA web-query study; this protocol alone does not establish broad cross-domain generalization, novelty, or state of the art. No new human labels are collected; use the benchmark qrels only.

Primary hypotheses on the APPS test split are that each of (1) E5+HyDE, (2) E5+reranking, and (3) E5+HyDE+reranking improves mean per-query `nDCG@10` over E5. Negative or null results are valid outcomes.

## Dataset and evaluation boundary

- Dataset: [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps/tree/ad8755dfa53f1de865e165d507fa0523502b5682), immutable revision `ad8755dfa53f1de865e165d507fa0523502b5682`.
- Use the `queries` configuration / `queries` split, `corpus` configuration / `corpus` split, and `default` qrels configuration. Use qrels `train` only for selection and `test` only for the final APPS evaluation.
- At this revision, the card declares 8,765 corpus records, 5,000 train qrels, and 3,765 test qrels. There is no development split. The CoIR paper identifies APPS as text-to-code contest retrieval: problem descriptions are queries and code solutions are documents.
- Input mapping is `queries.text` and `corpus.text`; omit `title`, `starter_code`, URL, and other metadata from model inputs. Use the official corpus and qrels without sampling. Candidate depth is 1,000.
- Create a 500-query selection subset from the 5,000 train query IDs before looking at scores: sort by the SHA256 digest of the prefix string `azh-526-apps-validation-v1`, one NUL byte, and the UTF-8 query ID; break digest ties by query ID and take the first 500. Use only those 500 qrels for fusion selection; leave the other 4,500 train queries unused. Save the selected IDs and their digest. The selection rule uses IDs only, not qrel labels.
- Freeze prompts, checkpoints, code, environment, and selected weights before opening test qrels or computing test metrics. Generate and save all four test rankings before evaluating any test qrels; compute the primary test metrics once for the complete matrix. Do not tune on test scores. If test results drive a method change, label the test exploratory and require a new untouched evaluation set for confirmation.
- Use the existing pinned runtime and `coir-eval==0.7.0` evaluator path. Preserve the current local ranking policy: descending score, saved insertion order for ties, then unique ordinal scores for evaluation. Report that tie policy and do not present the result as an unqualified reproduction of a stock CoIR leaderboard score.

The dataset card has no license declaration. The [upstream APPS repository](https://github.com/hendrycks/apps) identifies its repository license as MIT, which does not settle rights for every original contest problem or solution. Keep raw APPS records and generated code local; do not redistribute them. Before public release of data-bearing artifacts or excerpts, review the source URLs and applicable contest terms. This is a research handling constraint, not a legal determination.

## Checkpoints and fixed controls

| Component | Checkpoint and revision | Frozen use |
| --- | --- | --- |
| First-stage retriever | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2/tree/f52bf8ec8c7124536f0efb74aca902b2995e5bcd), `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | Existing E5 path; `query: ` / `passage: ` prefixes; 512 tokens; normalized embeddings; exact inner-product retrieval; FP32 CPU inference; batch 128; 8 CPU threads; seed 42. |
| HyDE generator | [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242), `c03e6d358207e414f1eca0bb1891e29f1db0e242` | One greedy hypothesis per query using the checkpoint's chat template and EOS stopping; no sampling, fallback prompt, or test-time prompt search. Load the released BF16 weights. |
| Pairwise reranker | [`NamanAgnih0tri/code-reranker-miniLM-staqc`](https://huggingface.co/NamanAgnih0tri/code-reranker-miniLM-staqc/tree/9bcf5f1cceef4914e2fc564b2e4ee78410626702), `9bcf5f1cceef4914e2fc564b2e4ee78410626702` | Sentence-Transformers `CrossEncoder` pair scoring of query text and candidate `corpus.text`; max length 512; higher score ranks first; FP32 GPU inference. Score only the E5 top-1,000 and preserve E5 order for score ties; never add documents. |

The reranker card reports StaQC SCA fine-tuning for Python question-to-code scoring, 85,294 training examples before negatives, and 170,588 pairs including negatives. Its pinned configuration declares a one-label sequence-classification head, and the card demonstrates `CrossEncoder.predict` pair scoring. The card also reports Conala results as self-reported. Treat the training description as the checkpoint publisher's claim, not independently audited provenance, and do not use its reported quality numbers as evidence here. The benchmark preflight must still verify that the pinned weights load and return one finite score per query/code pair; if they do not, stop and return for a protocol decision rather than silently substituting another model. The card declares Apache-2.0. The Qwen card declares Apache-2.0.

### HyDE prompt and representation

Use the Qwen chat template with `add_generation_prompt=True`:

```text
System: You are a code retrieval query expansion assistant. Write one concise Python 3 reference solution to the programming problem. Return code only. Do not restate the problem or explain the solution.
User: {query}
```

Set `do_sample=false`, `num_beams=1`, and `max_new_tokens=256`. Pass the full query up to the model's configured 32,768-token context; fail rather than truncate any longer query. Keep the original query. Build the transformed query from the original query, two LF characters, `Reference solution:`, one LF character, and the generated hypothesis, trimming only leading/trailing whitespace from the hypothesis. Store the exact generated text and token count. Empty output is a failed run, not a reason to use a fallback. The generator receives query text only, never qrels, reference answers, or corpus documents.

## System matrix and selection

| ID | APPS condition |
| --- | --- |
| `e5` | Original query with the fixed E5 first stage. |
| `e5_hyde` | Retrieve with the transformed query; fuse its ranking with original-query E5 using weighted RRF. Build the transformed query ranking from the union of the two top-1,000 lists, then truncate deterministically to 1,000. |
| `e5_rerank` | Pair-score and reorder only the original E5 top-1,000; fuse that order with E5 using weighted RRF. |
| `e5_hyde_rerank` | Weighted RRF of the selected `e5_hyde` and `e5_rerank` component rankings, truncated deterministically to 1,000. |

Use `rrf_k=60`. For each of the three fusion weights, evaluate the 21-point grid `0.00, 0.05, …, 1.00` on the 500-query selection subset only. When scores tie, select the larger weight on the first input: E5 for each single-component fusion and `e5_hyde` for the combined fusion. Freeze the selected weights and artifact digests before the APPS test pass. Keep raw query-plus-hypothesis and raw reranker orders as secondary diagnostics, never additional primary systems.

## Outcomes and statistics

- Primary outcome: mean per-query `nDCG@10`, using binary CoIR qrels and the evaluator/tie policy above. Report each system's score and the absolute delta versus E5 on the complete test set.
- Primary contrast family: the three two-sided paired comparisons of each treatment against E5. Use 100,000 paired sign-flip permutations per contrast, seed `20261014`, two-sided p-values with the plus-one correction, and Holm adjustment at family-wise `alpha=0.05`.
- Report paired-bootstrap percentile intervals from 10,000 query-level resamples. Use 98.33% intervals per contrast (Bonferroni family-wise 95%) and the mean paired delta. Preserve per-query metric rows and the resampling seed.
- All other metrics, raw component rankings, generated-hypothesis review, and latency/memory observations are secondary and descriptive. Do not make confirmatory claims from them.

## Compute, cost, and stopping rules

- Use one remote GPU with at least 24 GiB VRAM and a host with at least 32 GiB RAM. E5 remains FP32 on CPU with 8 threads; run Qwen in its released BF16 precision and the reranker in FP32 on that GPU. Record exact GPU/driver, CPU/RAM, package versions, precision, thread count, and device per stage.
- Hard ceiling: 48 billable GPU-hours total across preflight, selection, and the complete APPS evaluation. Use existing or no-cost/covered allocation only; this protocol does not authorize a paid purchase or charge. If the required allocation is unavailable or the preflight forecast exceeds the ceiling, stop before launching the complete run and request a revised budget. Do not shrink the test set, lower candidate depth, quantize, or change checkpoints to fit.
- Record dataset/model download and load time separately; measure end-to-end generation, embedding, retrieval, reranking, and evaluation time; report throughput, peak GPU memory, and peak host RAM. Do not omit failed or partial attempts from the run history.
- Run a real-model wiring preflight on selection-split queries only. A smoke result is not benchmark evidence. A complete result requires all 500 selection queries and all 3,765 APPS test queries, the complete 8,765-document corpus, all four systems, and resolved provenance. Any partial or failed execution remains partial/failed and cannot support a quality claim.

## Historical CosQA boundary

The existing CosQA protocol and artifacts remain governed by [`research-spec.md`](research-spec.md), [`reproducibility.md`](reproducibility.md), and [`experiment-provenance-audit.md`](experiment-provenance-audit.md). Its test split and chosen validation weights have already been examined. Preserve its original revisions and scores as retrospective evidence; do not relabel it prospective, retune from its test scores, overwrite prior outputs, or combine its results with APPS for the primary inference.

## Source record

- CoIR paper: https://aclanthology.org/2025.acl-long.1072.pdf
- APPS CoIR dataset at the pinned revision: https://huggingface.co/datasets/CoIR-Retrieval/apps/tree/ad8755dfa53f1de865e165d507fa0523502b5682
- APPS upstream repository and license notice: https://github.com/hendrycks/apps
- Qwen model card at the pinned revision: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242
- StaQC reranker card at the pinned revision: https://huggingface.co/NamanAgnih0tri/code-reranker-miniLM-staqc/tree/9bcf5f1cceef4914e2fc564b2e4ee78410626702
