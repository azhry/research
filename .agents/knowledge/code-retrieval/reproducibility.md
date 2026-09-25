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

### Baseline parity gate

Before interpreting an improvement, run the unchanged baseline through the new
pipeline and compare it with the saved baseline artifact. The baseline score,
query/corpus counts, evaluator, and configuration identity must match within a
pre-declared numeric tolerance. If baseline parity fails, stop: the result is not
an objective comparison and must not be reported as an improvement until the
configuration or cache problem is resolved.

### Cache and artifact rules

- Reuse a cache only when its dataset revision, model revisions, preprocessing,
  parameters, code/config identity, and execution semantics match the requested
  run.
- If candidate depth or ranking logic changes, regenerate the affected rankings
  even when embeddings can be reused safely.
- Keep corpus embeddings, query embeddings, generated HyDE text, first-stage
  rankings, and re-ranker scores separately identifiable.
- Record the cache identity, code commit, notebook hash, resolved model
  revisions, and configuration in the result artifact.
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

## Notebook order

1. Check Python, GPU, and dependencies.
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

## Evidence levels

- Smoke run: proves wiring on a tiny subset; not benchmark evidence.
- Regression test: proves deterministic adapters, shapes, cache invalidation, or
  metric calculations; not model-quality evidence.
- Benchmark run: complete declared split with real models and the official or
  validated evaluator; supports the reported `nDCG@10` result.

Never replace real model calls with fake LLM/Codex responses or synthetic scores, and
never report a failed or partial run as a completed benchmark.
