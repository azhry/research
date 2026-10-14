# Code Retrieval research project

## Objective

Reproduce and ablate the existing E5 code-retrieval study on CoIR CosQA. Measure how query representation, rank fusion, and reranking affect retrieval on this one fixed benchmark. The work is a focused reproducibility and ablation study; it does not establish cross-dataset generalization, novelty, or state of the art.

The CosQA test results and validation choices have already been examined. New test-split results are retrospective and exploratory, not independent confirmation. The exact experiment used in the conference submission remains unresolved.

## Study contract

| Item | Value |
| --- | --- |
| Repository | `https://github.com/azhry/research.git` |
| Directory | `code-retrieval` |
| Language/runtime | Python; existing Jupyter notebooks |
| Benchmark | CoIR |
| Dataset | `CoIR-Retrieval/cosqa`, revision `0846fa3b963a21bead36e9fab61451fc83b777a6` |
| Split | Complete `valid` (500 queries) and `test` (500 queries); full 20,604-document corpus |
| Retriever | `intfloat/e5-base-v2`, revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| HyDE generator | `google/flan-t5-base`, revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` |
| Existing reranker | `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| Code-trained comparator | `NamanAgnih0tri/code-reranker-miniLM-staqc`, revision `9bcf5f1cceef4914e2fc564b2e4ee78410626702`; planned, not yet run |
| Inference | Local CPU only; current settings use 8 PyTorch threads. No GPU or remote compute is assumed. |
| Evaluator | `coir-eval==0.7.0`, with the repository's saved-order tie policy |
| Primary metric | `nDCG@10` |

The complete experiment contract, model prompts, ablations, selection boundary, CPU preflight, and evidence limits are in [the CosQA protocol](study-protocol.md).

## Comparison scope

The existing four-system comparison remains the reproduction baseline:

1. E5 on the original query.
2. E5 on the original query plus one generated hypothesis, fused with the original E5 ranking.
3. The existing general-purpose cross-encoder reranking the E5 candidate pool, fused with the original E5 ranking.
4. Fusion of the selected HyDE and reranker rankings.

The focused additions are:

- E5 on the original query plus generated text without fusion.
- E5 on the generated hypothesis alone (canonical HyDE), using the same saved hypotheses.
- One code-trained reranker comparison using the same E5 top-1,000 candidate IDs, after its real pinned weights pass a CPU load and finite-score preflight.

These additions do not create a full generator-by-reranker factorial. The current four-system notebook is the starting point; keep implementation in that existing workflow.

## Claim and evidence boundary

Use the validation split for selection and preserve the full test split for descriptive evaluation. Never tune settings against test results. Because historical test values have already been reviewed, label all test comparisons as retrospective/exploratory. Do not pool results with another dataset or claim that this study validates general code search.

R1's offered CosQA-only path is selected. The code-oriented comparator is planned but has not been measured. R3's request for a second dataset and concern about Flan-T5-base hypothesis quality remain unresolved; this scope adds neither a second dataset nor a stronger-generator audit. State both limitations plainly.

## Deliverables

- The existing CosQA notebook and reusable modules, updated for the two representation ablations and code-trained reranker comparison.
- A machine-readable comparison table with complete per-query metrics and run provenance.
- Preserved generated hypotheses, rankings, fusion selections, model/data revisions, runtime and memory observations.
- A paper-ready interpretation that reports negative or null results and separates measured evidence from limitations.

## Execution controls

Run E5, Flan-T5-base, and reranker inference on CPU. The code-trained comparator must use its real pinned weights on CPU; if it cannot load, return finite pair scores, or fit available local resources, stop and report that limitation. Do not substitute another model, downsample the benchmark, quantize weights, or move execution to a remote GPU to complete this scope.

Preserve earlier run identities, caches, rankings, notebook outputs, and reported scores. Never fabricate model outputs or scores. Do not create duplicate one-off scripts or parallel notebooks.
