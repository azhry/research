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
Transformers 4.38.1. This project's reference runtime is pinned separately in
`.python-version` and `requirements.txt`; each run records its hardware, package
versions, precision settings, notebook hash, and commit.

The baseline retrieval follows the paper-faithful exact Faiss `IndexFlatIP`
path over normalized E5 embeddings with a 1,000-document candidate depth. The
actual runtime and package versions are persisted with each result; smoke runs
are wiring evidence only and are not comparable benchmark results.

Use Python 3.13.5 and install the pinned dependencies before opening a notebook:

```bash
python -m pip install -r code-retrieval/requirements.txt
```

Run benchmark notebooks through `scripts/run_benchmark_notebook.py` from the
`code-retrieval` directory. It checks the interpreter and every dependency
against the pins, then gives Jupyter a temporary kernel that points to that
same interpreter. This prevents Jupyter from silently running a notebook with a
different global Python installation.

The smoke notebook defaults to a real-model smoke run so that wiring can be
checked without encoding the full corpus. Smoke output is explicitly
non-benchmark evidence. The full notebook runs the complete CosQA test split
and corpus without needing `E5_BASELINE_MODE`. `E5_BASELINE_BATCH_SIZE` and
`E5_BASELINE_TORCH_THREADS` are explicit execution controls. Benchmark defaults
are batch size 128 and 8 PyTorch threads; the selected values are written to the
artifact and cache identity. All primary experiments use CPU inference and the
same 1,000-document candidate depth.

```bash
E5_BASELINE_MODE=benchmark E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_baseline_experiment.ipynb --output e5_baseline_full.executed.ipynb --output-dir artifacts/e5_baseline
```

To execute the full notebook directly from the `code-retrieval` directory:

```bash
E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_baseline_full_experiment.ipynb --output e5_baseline_full.executed.ipynb --output-dir artifacts/e5_baseline_full
```

On PowerShell, use `$env:E5_BASELINE_MODE='benchmark'` in the session before the
same command. Smoke output is written below
`code-retrieval/artifacts/e5_baseline/`; full-run
output is written below `code-retrieval/artifacts/e5_baseline_full/`. Both are
ignored by Git. Each run is also kept under `runs/<cache_identity>/`; the root
`result.json` and `metadata.json` files show the latest run.

`notebooks/e5_hyde_rerank_experiment.ipynb` runs the controlled four-system
comparison required by the study: E5, E5 + HyDE, E5 + re-ranking, and E5 +
HyDE + re-ranking. It uses the same CosQA data, paper-faithful E5
`IndexFlatIP` retrieval at candidate depth 1,000, qrels, and `nDCG@10`
evaluator for every row. HyDE uses the pinned `google/flan-t5-base` generator
and keeps the original programming question alongside its hypothesis. The
cross-encoder ranking is fused with E5 rather than replacing it. Fusion weights
are selected on the complete `valid` qrels split, recorded with the run, and
applied once to held-out `test`. Raw component scores remain diagnostics. The
combined row fuses the selected E5+HyDE and E5+reranker rankings. Resolved model
revisions, fusion controls, and component rankings are saved with each run.
Smoke output is wiring evidence only.

```bash
E5_HYDE_RERANK_MODE=benchmark E5_HYDE_RERANK_BATCH_SIZE=128 E5_HYDE_RERANK_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_hyde_rerank_experiment.ipynb --output e5_hyde_rerank_full.executed.ipynb --output-dir artifacts/e5_hyde_rerank
```

Comparison output is written below `code-retrieval/artifacts/e5_hyde_rerank/`;
each run is preserved under `runs/<cache_identity>/` and the directory is
ignored by Git.

## E5 + rerank

`notebooks/e5_rerank_experiment.ipynb` reuses the paper-faithful Faiss E5
first-stage contract and scores its top-1000 candidate pool with
`cross-encoder/ms-marco-MiniLM-L6-v2`. It evaluates the raw reranker as a
diagnostic and reports a validation-selected RRF of E5 and cross-encoder orders
as the experiment score. Both outputs use the same CosQA qrels and official
COIR evaluator. The reranker revision, fusion weight, max length, batch size,
device, seed, cache identity, timings, and provenance are recorded.

The default is a real-model smoke run and is not benchmark evidence:

```bash
python scripts/run_benchmark_notebook.py notebooks/e5_rerank_experiment.ipynb --output e5_rerank_smoke.executed.ipynb --output-dir artifacts/e5_rerank
```

Run the complete comparison with:

```bash
E5_RERANK_MODE=benchmark E5_RERANK_E5_BATCH_SIZE=128 E5_RERANK_BATCH_SIZE=128 E5_RERANK_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_rerank_experiment.ipynb --output e5_rerank_full.executed.ipynb --output-dir artifacts/e5_rerank
```

On PowerShell, set the variables with `$env:E5_RERANK_MODE='benchmark'` and
the corresponding `$env:` assignments before the same command. Smoke and
full-run output is written below `code-retrieval/artifacts/e5_rerank/`, which is
ignored by Git. Benchmark evidence is valid only when the notebook uses the
complete declared query and corpus populations with real E5 and cross-encoder
inference.

## E5 + HyDE

`notebooks/e5_hyde_experiment.ipynb` adds one real, deterministic HyDE
hypothesis from the pinned `google/flan-t5-base` revision to each original
CosQA test query, then
uses the same paper-faithful E5 first stage, full corpus, 1,000-document
candidate depth, qrels, and COIR evaluator as the baseline. It uses the same
prompt, deterministic generation settings, and fallback prompts as the combined
experiment. The E5 query text is retained alongside generated text, and the
reported score is validation-selected RRF of the original E5 and transformed
rankings. Raw HyDE and baseline scores are included as diagnostics. Controls,
resolved revision, and both fused and raw rankings are persisted. If all fallback
prompts produce empty output, execution stops without a score. HyDE does not
receive answers, qrels, labels, or target documents.

The smoke notebook run is wiring evidence only:

```bash
cd code-retrieval
E5_HYDE_MODE=smoke python scripts/run_benchmark_notebook.py notebooks/e5_hyde_experiment.ipynb --output e5_hyde_smoke.executed.ipynb --output-dir artifacts/e5_hyde
```

Run the complete declared benchmark only when model and dataset downloads and
available memory are sufficient:

```bash
cd code-retrieval
E5_HYDE_MODE=benchmark E5_HYDE_BATCH_SIZE=128 E5_HYDE_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_hyde_experiment.ipynb --output e5_hyde_full.executed.ipynb --output-dir artifacts/e5_hyde
```

HyDE output is written below `code-retrieval/artifacts/e5_hyde/`; each run is
preserved under `runs/<cache_identity>/`. Generated notebooks, caches, results,
and metadata are ignored by Git.

Select fusion weights only from the complete `valid` split. Run both notebooks
in benchmark mode with validation qrels, then select weights:

```bash
E5_HYDE_MODE=benchmark E5_HYDE_QRELS_SPLIT=valid E5_HYDE_BATCH_SIZE=128 E5_HYDE_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_hyde_experiment.ipynb --output e5_hyde_valid.executed.ipynb --output-dir artifacts/e5_hyde/validation
E5_RERANK_MODE=benchmark E5_RERANK_QRELS_SPLIT=valid E5_RERANK_E5_BATCH_SIZE=128 E5_RERANK_BATCH_SIZE=128 E5_RERANK_TORCH_THREADS=8 python scripts/run_benchmark_notebook.py notebooks/e5_rerank_experiment.ipynb --output e5_rerank_valid.executed.ipynb --output-dir artifacts/e5_rerank/validation
python scripts/select_fusion_weights.py
```

The selector validates complete
depth-1,000 rankings, sweeps E5/component weights in 0.05 increments, records
all validation scores and input hashes, and writes the selected settings below
`artifacts/e5_hyde_rerank/validation/weight_selection.json`. It reads no test
qrels. Then run `python scripts/verify_test_hyde_rankings.py`; it recomputes and
checks the held-out query-plus-HyDE rankings without loading test qrels. Finally,
run `python scripts/assemble_test_fusion_results.py` to apply those fixed
weights to the complete test split. The assembler verifies ranking provenance,
scores all systems using the same saved-order tie policy, and re-evaluates the
historical fixed system for a comparable reference before writing the artifact.

The official COIR evaluator receives unique ordinal scores that preserve each
artifact's descending score order and saved insertion order for ties. This
avoids pytrec_eval treating equal-score documents as unordered ties in one
system while RRF turns those same ranks into unique scores in another.

Keep the project's `.venv` active for the selector, ranking verifier, and test
assembler as well; each script rejects a Python or dependency version that
does not match the pinned runtime.
