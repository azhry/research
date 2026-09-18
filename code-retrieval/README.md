# E5 baseline

`notebooks/e5_baseline_experiment.ipynb` is the smoke-friendly entry point for
the CosQA E5-base-v2 baseline. The full-dataset entry point is
`notebooks/e5_baseline_full_experiment.ipynb`, which hard-codes benchmark mode
and refuses to write a result if a subset is selected. Both notebooks load the
pinned Hugging Face dataset revision, use text-only `query: ` / `passage: `
inputs, and evaluate with the COIR evaluator at `nDCG@10`.

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
same command. Generated caches, results, and metadata are written below
Smoke output is written below `code-retrieval/artifacts/e5_baseline/`; full-run
output is written below `code-retrieval/artifacts/e5_baseline_full/`. Both are
ignored by Git.

`notebooks/e5_hyde_rerank_experiment.ipynb` runs the controlled four-system
comparison required by the study: E5, E5 + HyDE, E5 + re-ranking, and E5 +
HyDE + re-ranking. It uses the same CosQA data, E5 first-stage candidate depth,
qrels, and `nDCG@10` evaluator for every row. HyDE uses the local
`google/flan-t5-base` generator, and the re-ranker uses
`cross-encoder/ms-marco-MiniLM-L6-v2`; their resolved revisions and generation
controls are recorded in the result metadata. The cross-encoder can only reorder
the E5 candidate IDs. Smoke output is wiring evidence only.

```bash
E5_HYDE_RERANK_MODE=benchmark E5_HYDE_RERANK_BATCH_SIZE=128 E5_HYDE_RERANK_TORCH_THREADS=8 python -m nbconvert --to notebook --execute code-retrieval/notebooks/e5_hyde_rerank_experiment.ipynb --output e5_hyde_rerank_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

Comparison output is written below `code-retrieval/artifacts/e5_hyde_rerank/`;
the directory is ignored by Git.
