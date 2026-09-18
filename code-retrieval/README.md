# E5 baseline

`notebooks/e5_baseline_experiment.ipynb` is the smoke-friendly entry point for
the CosQA E5-base-v2 baseline. The full-dataset entry point is
`notebooks/e5_baseline_full_experiment.ipynb`, which hard-codes benchmark mode
and refuses to write a result if a subset is selected. Both notebooks load the
pinned Hugging Face dataset revision, use text-only `query: ` / `passage: `
inputs, and evaluate with the COIR evaluator at `nDCG@10`.
The baseline retrieval depth is 1000, matching the official COIR evaluator's
default candidate depth; the primary reported metric remains `nDCG@10`.

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

## E5 + rerank

`notebooks/e5_rerank_experiment.ipynb` reuses the official COIR E5 first-stage
contract and scores its top-1000 candidate pool with
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
