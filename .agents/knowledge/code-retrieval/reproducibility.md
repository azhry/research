# Code Retrieval reproducibility rules

## Objective comparison contract

Treat two benchmark runs as directly comparable only when every control below is
identical. A rerun should start from the saved baseline configuration and change
only the explicitly requested experimental variable.

### Controls that must remain fixed

- **Dataset:** dataset name, immutable revision or snapshot, corpus and query
  splits, qrels split, field mapping, corpus text/title policy, query filtering,
  and the complete query set. Record the query count, corpus count, and any
  exclusions.
- **Baseline model:** model identifier, immutable weight/tokenizer revision,
  pooling behavior, query and passage prefixes, truncation/max sequence length,
  normalization, and device/runtime settings that can affect numerical results.
  Every system in the comparison must use the same first-stage baseline model.
- **Retrieval parameters:** candidate depth, similarity function, index/search
  settings, tie-breaking, and batch/configuration values. Re-ranking must receive
  the same first-stage candidate pool and must not add documents outside it.
- **Transformation parameters:** for HyDE or another query transformation, keep
  generator model/revision, prompt, hypothesis count, temperature/sampling,
  maximum tokens, stopping behavior, combination strategy, and fallback behavior
  fixed. The transformation may change only the query representation.
- **Re-ranker parameters:** model identifier/revision, input formatting, maximum
  length, score direction, candidate depth, and tie-breaking must remain fixed.
- **Evaluation:** evaluator implementation and package version, metric and cutoff
  (for this study, COIR `nDCG@10`), qrels interpretation, aggregation, and
  relevance judgments must be identical.
- **Execution controls:** seed, deterministic settings, precision, dependency
  versions, CPU/GPU details, and thread counts should remain fixed and must be
  recorded. Batch size may be changed only when it is proven not to alter model
  outputs; otherwise it defines a separate run.
- **Notebook kernel:** benchmark notebooks must execute with the interpreter
  and pinned package versions declared by `code-retrieval/.python-version` and
  `requirements.txt`. Jupyter's default `python3` kernel can point at a
  different global installation even when `nbconvert` was launched from the
  project's virtual environment. Use `scripts/run_benchmark_notebook.py`,
  which verifies the pins and explicitly selects the current interpreter as
  the notebook kernel. A run from another environment is exploratory until
  rerun under the pinned environment.

### Baseline parity gate

Before interpreting an improvement, run the unchanged baseline through the new
pipeline and compare it with the saved baseline artifact. The baseline score,
query/corpus counts, evaluator, and configuration identity must match within a
pre-declared numeric tolerance. If baseline parity fails, stop: the result is not
an objective comparison and must not be reported as an improvement until the
configuration or cache problem is resolved.

### Cache and artifact rules

- Reuse a cache only when its dataset revision, model revisions, preprocessing,
  parameters, code/config identity, notebook hash, runtime versions, device,
  precision, thread count, and execution semantics match the requested run.
- If candidate depth or ranking logic changes, regenerate the affected rankings
  even when embeddings can be reused safely.
- Keep corpus embeddings, query embeddings, generated HyDE text, first-stage
  rankings, and re-ranker scores separately identifiable.
- Ranking caches must preserve the original retrieval order, including the order
  of tied scores. Do not alphabetically sort document IDs in a ranking map.
- Record the cache identity, code commit, notebook hash, resolved model
  revisions, and configuration in the result artifact.
- Preserve every run under its cache identity. A root-level latest result may be
  convenient, but it must not replace prior run artifacts.
- A cache with unknown, stale, or mismatched provenance is not primary evidence.

### Reporting rules

Report the complete system matrix, not only the best variant: the unchanged
baseline, each individual component, and the combined system. Include the absolute
score and absolute delta versus the baseline, plus query/corpus counts, candidate
and re-ranking depth, evaluator/version, and evidence level. Do not use a relative
percentage alone as the comparison.

If any control differs, label the run exploratory or create a new baseline; do not
call it a direct rerun. Never use synthetic scores, mocked model calls, partial
runs, or failed runs as benchmark evidence.

Candidate depths are part of the retrieval contract. Scores from depth 10, 100,
and 1000 runs are not directly comparable because Faiss can return a different
ordering for tied scores when the requested top-k changes. The primary comparison
uses depth 1000 for E5, HyDE, and both re-ranked systems.

The COIR evaluator groups documents that have exactly equal scores. The saved
retrieval artifacts also preserve a deterministic order for equal-score
documents, so evaluating raw similarities could report a different metric from
an RRF ranking built from that same order. The primary evaluator now sorts by
descending source score with saved insertion order as the tie-break, assigns
unique ordinal scores, then calls the pinned COIR evaluator. This makes a
one-source RRF exactly match the source ranking and keeps every system's metric
based on its recorded order. Older raw-score metrics must be labeled with their
tie policy before comparing them to current results.

## Validation and method selection

Use the CosQA `valid` qrels split to choose query-combination and fusion settings.
Keep the `test` qrels split held out until those settings are fixed. Record the
validation split, complete query count, candidate depth, evaluator version, every
tested setting, and the selected setting in the run artifacts. A test result may
confirm the selected method once; do not search test results for a better weight.
For weighted RRF, sweep the E5 weight from 0.00 through 1.00 in 0.05 steps with
`rrf_k=60`; when validation scores tie, select the higher E5 weight.

The current HyDE representation retains the original programming question and
appends its generated hypothesis. The experiment reports both the raw transformed
ranking and a weighted reciprocal-rank fusion of that ranking with the original
E5 ranking. The reranker experiment likewise reports its raw cross-encoder order
as a diagnostic and uses validation-selected fusion with the unchanged E5 order
for its comparison score. This rerank fusion must preserve the E5 candidate IDs
exactly. HyDE fusion can draw from the union of the two depth-1000 retrieval lists;
record the ordered output IDs and their digest, then truncate deterministically to
1,000. The combined system fuses the two selected component rankings and also
records its resulting IDs. Save every fused ranking and its weights with the run
identity.

## Historical artifact differences

Do not average or directly compare the older outputs that used candidate depths
10, 100, and 1,000: Faiss tie order can change with the requested depth. Some
older reranker artifacts also came from a different Python, PyTorch,
Transformers, and NumPy environment. The standalone reranker previously broke
equal cross-encoder scores by document ID, while the combined runner retained
first-stage order; both now retain E5's order for tied scores. Earlier HyDE
outputs also differed because one run replaced the question with generated text
and another retained it. Current HyDE runs retain the question, and any ranking
or runtime difference stays isolated under its own cache identity.

The artifact audit found these concrete examples. The older full baseline used
depth 10 and scored 0.28082. The standalone depth-1,000 reranker used Python
3.12.10, PyTorch 2.11.0, Transformers 4.57.6, and NumPy 2.4.2; it scored 0.20725
against its E5 score of 0.33104. The pinned depth-1,000 combined run used Python
3.13.5, PyTorch 2.14.0, Transformers 5.17.0, and NumPy 2.5.3; its raw
hypothesis-only HyDE and cross-encoder scores were 0.14081 and 0.20736. These
numbers describe different controls and must not be treated as one score trend.

With the order-preserving evaluator, the complete valid split scores 0.44119 for
E5, 0.41717 for raw query-plus-hypothesis retrieval, and 0.45683 for the
validation-selected 0.65/0.35 E5/HyDE RRF. The raw CrossEncoder order scores
0.27517; its selected 0.80/0.20 E5/RRF scores 0.45177. The combined selected
0.75/0.25 component fusion scores 0.45932. The weight sweep covers all 21 E5
weights from 0.00 to 1.00 in 0.05 increments over all 500 valid queries.

The complete held-out test comparison, using those fixed validation-selected
weights once, scores 0.45876 for E5, 0.47292 for E5+HyDE, 0.46085 for E5 plus
re-ranking, and 0.47821 for the combined system. Their absolute deltas versus
E5 are +0.01416, +0.00209, and +0.01945. Raw query-plus-hypothesis and raw
CrossEncoder diagnostics score 0.41394 and 0.28619. The saved historical
four-source result is reproduced at 0.46418 using its original hypothesis-only
representation and 0.90/0.05/0.04/0.01 weights; the selected current combined
system is +0.01403 higher. This is a useful fixed-result comparison, while the
underlying system designs still differ.

Older direct-score baselines such as 0.32107 on valid and 0.33104 on test are
not improvements or regressions against the order-preserving baseline. Their
same saved E5 rankings re-evaluate to 0.44119 and 0.45876 when deterministic
similarity ties are treated as the stored Faiss order. The older raw-component
scores likewise use the previous tie policy. Keep those values in the artifact
history, but use the current policy for every new system comparison.

## Notebook order

1. Check Python, GPU, and dependencies; execute through the pinned-kernel
   runner for benchmark evidence.
2. Capture versions, model revisions, dataset revision, and seed.
3. Load data and inspect the actual schema.
4. Encode/cache the corpus.
5. Run and evaluate E5.
6. Generate/cache HyDE text and evaluate E5 + HyDE.
7. Re-rank the shared candidate pools for the two re-ranked systems.
8. Produce the comparison table, artifacts, and limitations.

Keep reusable logic in modules once it exceeds a few lines; keep the notebook as the
ordered experiment entry point. Avoid hidden state or manually edited result cells.

## Cache validity

Cache corpus embeddings, HyDE generations, first-stage rankings, and re-ranker scores
separately. A cache is valid only when dataset/model revisions, preprocessing,
parameters, and code/config identity match. Unknown-provenance caches are not valid
primary evidence.

## Execution and analysis traces

Run benchmark notebooks through `scripts/run_benchmark_notebook.py`. Every
invocation preserves a timestamped run directory below its `--output-dir` with
the executed notebook, readable `execution.log`, structured
`execution_trace.jsonl`, and `execution.json`. The trace records notebook
start/end, per-code-cell start/end, section, source hash, elapsed seconds,
status, textual outputs, and exceptions. Metadata records the source notebook
hash, commit, pinned Python/runtime, safe execution controls, result/cache
identity, control checks, and exact artifact paths. A failed run keeps available
diagnostics and returns non-zero; it is never promoted to benchmark evidence.

For complete benchmark results, the runner exports `per_query_metrics.csv` and
`qualitative_cases.jsonl` from the saved rankings and pinned qrels. The metrics
include per-query nDCG@10, delta versus E5, relevant-document counts, retrieval
rank observations, and ranking hashes. Qualitative cases preserve query text,
available HyDE text, top-10 document IDs/scores/relevance, and bounded document
text previews. The arithmetic mean of each per-query metric must match the
reported official aggregate within `1e-5`; otherwise export fails. These
per-query calculations are analysis outputs, not new rankings or synthetic
benchmark scores.

The published CoIR Table 3 E5-base/CosQA value is `0.3259` nDCG@10. It may be
reported as the issue's external paper reference, with protocol differences
disclosed. It is not a substitute for the unchanged local baseline-parity gate.

## Evidence levels

- Smoke run: proves wiring on a tiny subset; not benchmark evidence.
- Regression test: proves deterministic adapters, shapes, cache invalidation, or
  metric calculations; not model-quality evidence.
- Benchmark run: complete declared split with real models and the official or
  validated evaluator; supports the reported `nDCG@10` result.

Never replace real model calls with fake LLM/Codex responses or synthetic scores, and
never report a failed or partial run as a completed benchmark.
