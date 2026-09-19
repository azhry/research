# E5 baseline

`notebooks/e5_baseline_experiment.ipynb` is the smoke-friendly entry point for
the CosQA E5-base-v2 baseline. The full-dataset entry point is
`notebooks/e5_baseline_full_experiment.ipynb`, which hard-codes benchmark mode
and refuses to write a result if a subset is selected. Both notebooks load the
pinned Hugging Face dataset revision, use text-only `query: ` / `passage: `
inputs, and evaluate with the COIR evaluator at `nDCG@10`.

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
same command. Generated caches, results, and metadata are written below
Smoke output is written below `code-retrieval/artifacts/e5_baseline/`; full-run
output is written below `code-retrieval/artifacts/e5_baseline_full/`. Both are
ignored by Git.

## E5 + HyDE

`notebooks/e5_hyde_experiment.ipynb` adds one real, deterministic HyDE
hypothesis from `google/flan-t5-base` to each original CosQA test query, then
uses the same paper-faithful E5 first stage, full corpus, 1,000-document
candidate depth, qrels, and COIR evaluator as the baseline. The generator
revision, prompt, generation settings, expanded-query strategy, and separate
cache identity are persisted with the result. HyDE does not receive answers,
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
