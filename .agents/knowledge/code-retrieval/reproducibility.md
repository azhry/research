# Code Retrieval reproducibility rules

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
