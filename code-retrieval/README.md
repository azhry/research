# E5 baseline

`notebooks/e5_baseline_experiment.ipynb` is the ordered entry point for the
CosQA E5-base-v2 baseline. It loads the pinned Hugging Face dataset revision,
inspects the observed schema, uses text-only `query: ` / `passage: ` inputs, and
evaluates the complete test qrels with the COIR evaluator at `nDCG@10`.

Install the declared dependencies before opening the notebook:

```bash
python -m pip install -r code-retrieval/requirements.txt
```

The notebook defaults to a real-model smoke run so that wiring can be checked
without encoding the full corpus. Smoke output is explicitly non-benchmark
evidence. Set `E5_BASELINE_MODE=benchmark` before executing to run the complete
CosQA test split and corpus. `E5_BASELINE_BATCH_SIZE` and
`E5_BASELINE_TORCH_THREADS` are explicit performance controls; the selected
values are written to metadata and cache identity.

```bash
E5_BASELINE_MODE=benchmark E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute code-retrieval/notebooks/e5_baseline_experiment.ipynb --output e5_baseline_full.executed.ipynb --ExecutePreprocessor.timeout=0
```

On PowerShell, use `$env:E5_BASELINE_MODE='benchmark'` in the session before the
same command. Generated caches, results, and metadata are written below
`code-retrieval/artifacts/e5_baseline/` and are ignored by Git.
