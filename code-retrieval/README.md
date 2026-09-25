# E5 baseline

`notebooks/e5_baseline_experiment.ipynb` is the smoke-friendly entry point for
the CosQA E5-base-v2 baseline. The full-dataset entry point is
`notebooks/e5_baseline_full_experiment.ipynb`, which hard-codes benchmark mode
and refuses to write a result if a subset is selected. Both notebooks load the
pinned Hugging Face dataset revision, use text-only `query: ` / `passage: `
inputs, and evaluate with the COIR evaluator at `nDCG@10`.
The baseline uses the paper-faithful exact Faiss `IndexFlatIP` inner-product
path over normalized E5 embeddings with a 1,000-document first-stage candidate
depth; the primary reported metric remains `nDCG@10`.

The paper-compatible controls are E5-base-v2, 512-token inputs, mean-pooled
embeddings, cosine/inner-product ranking, Faiss `IndexFlat`, and a 1,000-result
retrieval depth. The paper's historical environment was PyTorch 2.0.1 and
Transformers 4.38.1; every actual run records its installed versions in
`metadata.json` so a newer local runtime is not mistaken for an exact
environment reproduction.

The baseline retrieval follows the paper-faithful exact Faiss `IndexFlatIP`
path over normalized E5 embeddings with a 1,000-document candidate depth. The
actual runtime and package versions are persisted with each result; smoke runs
are wiring evidence only and are not comparable benchmark results.

Install the declared dependencies before opening the notebook:

```bash
python -m pip install -r code-retrieval/requirements.txt
```

The smoke notebook defaults to a real-model smoke run so that wiring can be
checked without encoding the full corpus. Smoke output is explicitly
non-benchmark evidence. The full notebook runs the complete CosQA test split
and corpus without needing `E5_BASELINE_MODE`. `E5_BASELINE_BATCH_SIZE` and
`E5_BASELINE_TORCH_THREADS` are explicit performance controls; the selected
values are written to metadata and cache identity.

```bash
E5_BASELINE_MODE=benchmark E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute code-retrieval/notebooks/e5_baseline_experiment.ipynb --output e5_baseline_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

To execute the full notebook directly from the `code-retrieval` directory:

```bash
E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute notebooks/e5_baseline_full_experiment.ipynb --output e5_baseline_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

On PowerShell, use `$env:E5_BASELINE_MODE='benchmark'` in the session before the
same command. Smoke output is written below
`code-retrieval/artifacts/e5_baseline/`; full-run
output is written below `code-retrieval/artifacts/e5_baseline_full/`. Both are
ignored by Git.

`notebooks/e5_hyde_rerank_experiment.ipynb` runs the controlled four-system
comparison required by the study: E5, E5 + HyDE, E5 + re-ranking, and E5 +
HyDE + re-ranking. It uses the same CosQA data, paper-faithful E5
`IndexFlatIP` first-stage retrieval at candidate depth 1,000 by default,
qrels, and `nDCG@10` evaluator for every row. HyDE uses the local
`google/flan-t5-base` generator, and the re-ranker uses
`cross-encoder/ms-marco-MiniLM-L6-v2`; their resolved revisions and generation
controls are recorded in the result metadata. The combined row uses a fixed,
E5-anchored weighted reciprocal-rank fusion of the original E5, HyDE, and both
cross-encoder rankings (`rrf_k=60`, weights `0.90/0.05/0.04/0.01`) and retains
the configured fused depth. Re-ranking is batched across the complete
candidate collection so the full run does not change its candidate contract.
Smoke output is wiring evidence only. Set
`E5_HYDE_RERANK_CANDIDATE_DEPTH=10` when comparing directly with the saved
E5 full-baseline artifact, which used candidate depth 10.

```bash
E5_HYDE_RERANK_MODE=benchmark E5_HYDE_RERANK_BATCH_SIZE=128 E5_HYDE_RERANK_TORCH_THREADS=8 python -m nbconvert --to notebook --execute code-retrieval/notebooks/e5_hyde_rerank_experiment.ipynb --output e5_hyde_rerank_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

Comparison output is written below `code-retrieval/artifacts/e5_hyde_rerank/`;
the directory is ignored by Git.

## E5 + rerank

`notebooks/e5_rerank_experiment.ipynb` reuses the paper-faithful Faiss E5
first-stage contract and scores its top-1000 candidate pool with
`cross-encoder/ms-marco-MiniLM-L6-v2`. It evaluates the original E5 ranking and
the reranked ranking with the same CosQA qrels and official COIR evaluator, then
writes both `nDCG@10` values and their delta to
`artifacts/e5_rerank/result.json`. The reranker revision, max length, batch
size, device, seed, cache identity, timings, and provenance are recorded.

The default is a real-model smoke run and is not benchmark evidence:

```bash
python -m nbconvert --to notebook --execute code-retrieval/notebooks/e5_rerank_experiment.ipynb --output e5_rerank_smoke.executed.ipynb --ExecutePreprocessor.timeout=0
```

Run the complete comparison with:

```bash
E5_RERANK_MODE=benchmark E5_RERANK_E5_BATCH_SIZE=128 E5_RERANK_BATCH_SIZE=128 E5_RERANK_TORCH_THREADS=8 python -m nbconvert --to notebook --execute code-retrieval/notebooks/e5_rerank_experiment.ipynb --output e5_rerank_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

On PowerShell, set the variables with `$env:E5_RERANK_MODE='benchmark'` and
the corresponding `$env:` assignments before the same command. Smoke and
full-run output is written below `code-retrieval/artifacts/e5_rerank/`, which is
ignored by Git. Benchmark evidence is valid only when the notebook uses the
complete declared query and corpus populations with real E5 and cross-encoder
inference.

## E5 + HyDE

`notebooks/e5_hyde_experiment.ipynb` adds one real, deterministic HyDE
hypothesis from `google/flan-t5-base` to each original CosQA test query, then
uses the same paper-faithful E5 first stage, full corpus, 1,000-document
candidate depth, qrels, and COIR evaluator as the baseline. The generator
revision, prompt, generation settings, expanded-query strategy, empty-output
policy, and separate cache identity are persisted with the result. If the
generator emits only special tokens, the declared `original_query_fallback`
policy keeps that query's representation equal to the original query; it does
not invent a hypothesis or use evaluation data. HyDE does not receive answers,
qrels, labels, or target documents.

The smoke notebook run is wiring evidence only:

```bash
cd code-retrieval
E5_HYDE_MODE=smoke python -m nbconvert --to notebook --execute notebooks/e5_hyde_experiment.ipynb --output e5_hyde_smoke.executed.ipynb --ExecutePreprocessor.timeout=0
```

Run the complete declared benchmark only when model and dataset downloads and
available memory are sufficient:

```bash
cd code-retrieval
E5_HYDE_MODE=benchmark E5_HYDE_BATCH_SIZE=128 E5_HYDE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute notebooks/e5_hyde_experiment.ipynb --output e5_hyde_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

HyDE output is written below `code-retrieval/artifacts/e5_hyde/`; generated
notebooks, caches, results, and metadata are ignored by Git.
