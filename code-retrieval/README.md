# CosQA code retrieval experiments

> This project studies CoIR CosQA only. The active work is a reproduction and
> ablation of the existing E5, Flan-T5-base HyDE, and general cross-encoder
> workflow. The study specification is
> [here](../.agents/knowledge/code-retrieval/research-spec.md).
> APPS, Qwen, a second dataset, and remote-GPU execution are outside this scope.
> CosQA test results have already been examined, so new test results are
> retrospective and exploratory, not independent confirmation.

## Study scope

The existing four-system notebook is the reproduction baseline. The specification adds
two representation diagnostics—query plus generated text without fusion, and
the same generated text alone—plus independent E5 RRF sweeps for both transformed
representations. Its opt-in code-reranker path adds one code-trained reranker on the
same E5 candidate IDs without changing the original four conditions. It pins
NamanAgnih0tri/code-reranker-miniLM-staqc at revision
9bcf5f1cceef4914e2fc564b2e4ee78410626702, uses real CPU CrossEncoder query/code
pairs, and requires a real-weight finite-score preflight before complete runs.
The path validates and reuses checksummed baseline rankings, selects the code
fusion weight on all 500 valid queries, and applies only that frozen weight to
the 500 test queries.
Completed evidence remains in its preserved artifact directory; new code-reranker
runs default to `artifacts/code_reranker/`. Preflight outputs are
operational evidence, and test outputs remain retrospective and exploratory.

The R1 CosQA-only alternative is selected. R3's second-dataset request and
Flan-T5-base quality concern remain unresolved and must be stated as limitations.
The study makes no cross-dataset generalization claim and does not benchmark a
stronger generator.

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

Activate the project's pinned `.venv` from the `code-retrieval` directory and
register its kernel once:

```bash
python -m ipykernel install --user --name code-retrieval --display-name code-retrieval
```

Execute an existing notebook directly with Jupyter, selecting that kernel.
Each notebook checks the Python and package pins before model inference, so a
global or otherwise mismatched kernel fails before producing benchmark results.
For example, from `code-retrieval`:

```bash
python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_hyde_rerank_experiment.ipynb --output e5_hyde_rerank.executed.ipynb --output-dir artifacts/runs/20260928T111402089Z_full-rerun
```

Keep `python` bound to the active `.venv` so both nbconvert and the kernel use
the declared environment. Choose a new `--output-dir` for each run.

The smoke notebook defaults to a real-model smoke run so that wiring can be
checked without encoding the full corpus. Smoke output is explicitly
non-benchmark evidence. The full notebook runs the complete CosQA test split
and corpus without needing `E5_BASELINE_MODE`. `E5_BASELINE_BATCH_SIZE` and
`E5_BASELINE_TORCH_THREADS` are explicit execution controls. Benchmark defaults
are batch size 128 and 8 PyTorch threads; the selected values are written to the
artifact and cache identity. All primary experiments use CPU inference and the
same 1,000-document candidate depth.

```bash
E5_BASELINE_MODE=benchmark E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_baseline_experiment.ipynb --output e5_baseline_full.executed.ipynb --output-dir artifacts/e5_baseline
```

To execute the full notebook directly from the `code-retrieval` directory:

```bash
E5_BASELINE_BATCH_SIZE=128 E5_BASELINE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_baseline_full_experiment.ipynb --output e5_baseline_full.executed.ipynb --output-dir artifacts/e5_baseline_full
```

On PowerShell, use `$env:E5_BASELINE_MODE='benchmark'` in the session before the
same command. Smoke output is written below
`code-retrieval/artifacts/e5_baseline/`; full-run
output is written below `code-retrieval/artifacts/e5_baseline_full/`. Both are
ignored by Git. Caches are stored by cache identity, while the artifact
directory's `result.json` and `metadata.json` show the latest run.

`notebooks/e5_hyde_rerank_experiment.ipynb` is the ordered entry point for the
four-system comparison: E5, E5 + HyDE, E5 + re-ranking, and E5 + HyDE +
re-ranking. It uses the pinned CosQA corpus and query revision, depth-1,000
E5 retrieval, RRF with `k=60`, the official COIR evaluator, and `nDCG@10` as
the selection metric. The original comparison keeps the programming question
with its generated hypothesis; the paired ablation also embeds that same saved
hypothesis alone. The cross-encoder reorders the same E5 candidate pool. The
combined system fuses the two selected component rankings.

Run validation selection and test evaluation as two separate notebook
executions. From the `code-retrieval` directory, first use benchmark mode with
the complete `valid` split. Replace `RUNID` with a unique output suffix:

```bash
E5_HYDE_RERANK_MODE=benchmark E5_HYDE_RERANK_QRELS_SPLIT=valid E5_HYDE_RERANK_BATCH_SIZE=128 E5_HYDE_RERANK_TORCH_THREADS=8 E5_HYDE_RERANK_ARTIFACT_DIR=artifacts/representation_fusion E5_HYDE_RERANK_CACHE_DIR=artifacts/representation_fusion/cache python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_hyde_rerank_experiment.ipynb --output e5_hyde_rerank_selection.executed.ipynb --output-dir artifacts/representation_fusion/notebooks/selection-RUNID
```

The selection run evaluates all 21 weights from 0.00 through 1.00, in 0.05
steps, for E5 + query-plus-hypothesis, E5 + hypothesis-only, E5 + reranker, and
the combined ranking. Both HyDE representations use the same saved generation
for each query. Each sweep chooses by `nDCG@10` rounded to five decimals and
prefers the higher first-input weight on a tie. The notebook prints a unique
immutable manifest path and digest. Copy
that exact path for the next execution with the complete `test` split, and use
a new `RUNID` for its notebook-output directory:

```bash
E5_HYDE_RERANK_MODE=benchmark E5_HYDE_RERANK_QRELS_SPLIT=test E5_HYDE_RERANK_SELECTION_MANIFEST=artifacts/representation_fusion/selection_runs/SELECTIONID/selection_manifest.json E5_HYDE_RERANK_BATCH_SIZE=128 E5_HYDE_RERANK_TORCH_THREADS=8 E5_HYDE_RERANK_ARTIFACT_DIR=artifacts/representation_fusion E5_HYDE_RERANK_CACHE_DIR=artifacts/representation_fusion/cache python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_hyde_rerank_experiment.ipynb --output e5_hyde_rerank_test.executed.ipynb --output-dir artifacts/representation_fusion/notebooks/test-RUNID
```

For the code-reranker path, enable `E5_HYDE_RERANK_CODE_RERANKER=1` and set
`E5_HYDE_RERANK_RANKING_REPLAY=code_reranker`. Point
`E5_HYDE_RERANK_REPLAY_VALID_SELECTION_MANIFEST` at the complete baseline valid-selection manifest for
preflight and selection. Run the one-pair
preflight first, then complete validation selection, then test evaluation with
the generated code-reranker selection manifest:

```bash
E5_HYDE_RERANK_MODE=benchmark E5_HYDE_RERANK_QRELS_SPLIT=valid E5_HYDE_RERANK_CODE_RERANKER=1 E5_HYDE_RERANK_CODE_PREFLIGHT_ONLY=1 E5_HYDE_RERANK_RANKING_REPLAY=code_reranker E5_HYDE_RERANK_REPLAY_VALID_SELECTION_MANIFEST=artifacts/representation_fusion/selection_runs/VALID_SELECTION_RUNID/selection_manifest.json E5_HYDE_RERANK_CODE_BATCH_SIZE=128 E5_HYDE_RERANK_BATCH_SIZE=128 E5_HYDE_RERANK_TORCH_THREADS=8 E5_HYDE_RERANK_ARTIFACT_DIR=artifacts/code_reranker E5_HYDE_RERANK_CACHE_DIR=artifacts/code_reranker/cache python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_hyde_rerank_experiment.ipynb --output code_reranker_preflight.executed.ipynb --output-dir artifacts/code_reranker/notebooks/preflight-RUNID
```

After a passed preflight, remove `E5_HYDE_RERANK_CODE_PREFLIGHT_ONLY=1` and
use a new output directory for full validation selection. For the separate test
execution, change the qrels split to `test`, set
`E5_HYDE_RERANK_SELECTION_MANIFEST` to the generated code-reranker selection manifest. Set
`E5_HYDE_RERANK_REPLAY_TEST_RESULT` to the preserved complete test
result, and use another unique notebook-output directory. Keep batch size 128, eight
CPU threads, CPU float32 inference, max length 512, and the complete 500-query
split and 20,604-document corpus. These environment controls add only the code
reranker condition; the original combined system remains unchanged.

Test evaluation fails closed if the manifest digest, dataset/model controls,
runtime, source or notebook hashes, validation baseline parity, or valid/test
query-ID and normalized-text boundary fails. The manifest contains full valid
and test-population fingerprints, qrels/corpus digests, rankings and metadata
hashes, every sweep score, selected weights, and the declared evaluator/tie
policy. It does not store raw query text or test labels in the selection stage.
Baseline parity is checked against the audited `nDCG@10` references 0.44119 on
valid and 0.45876 on test, with absolute tolerance 0.00001.

For PowerShell, set the same variables with `$env:E5_HYDE_RERANK_MODE='benchmark'`
and `$env:E5_HYDE_RERANK_QRELS_SPLIT='valid'` or `'test'` before running the
`python -m nbconvert` command. The test results were already examined, so this
evaluation is retrospective and exploratory, not independent confirmation.
Smoke runs remain wiring evidence only.

The test comparison's `comparison.csv` retains its four headline systems. Each
test comparison writes `representation_ablation.csv` with original E5, raw
query-plus-hypothesis, raw hypothesis-only, and both validation-selected E5
fusions. Validation selection writes the same ablation files beside its
manifest. The run-scoped `representation_ablation_per_query.csv` stores paired
primary and secondary metrics and deltas for the same queries. `result.json` records the shared
generation hash, both representation hashes, and paths and hashes for raw and
fused rankings. Historical replay cannot supply the hypothesis-only
ranking, so representation selection requires a fresh complete valid run.

Each comparison run records stage-level cache load, compute, and save times in
`runs/<run_id>/runtime_costs.json` and writes normalized timing observations to
`runtime_costs.csv`. The report includes model initialization, corpus encoding,
Faiss index construction, query encoding, HyDE generation, search, reranking,
latency percentiles and throughput, end-to-end query cost, and sampled process
RSS. It records the device, precision, batch sizes, thread count, candidate
depth, corpus/query sizes, timing and warm-up policy, and CUDA peak counters
when a CUDA model device is used. The CSV uses one row per cache operation,
pipeline phase, or observed model/retrieval batch. The first uncached run
measures actual inference without a warm-up; a second run on the same identity
shows cache-load cost separately. Smoke results remain wiring evidence, not
benchmark evidence.

The notebook's tagged query-level and secondary-metric analyses are optional
and skipped by default. To run them, set `E5_HYDE_RERANK_RUN_POSTHOC=1` and
`E5_HYDE_RERANK_POSTHOC_ANALYSIS_RUN_DIR` to the saved analysis run directory.

An optional `E5_HYDE_RERANK_RANKING_REPLAY=historical` mode can replay the
preserved real-model ranking artifacts when `E5_HYDE_RERANK_RANKING_REPLAY_ROOT`
points to the historical `artifacts` directory. It verifies the saved file
hashes, pinned revisions, split, population, candidate contract, and original
run record. Replay performs no new model inference and is labeled
`cache_replay`; it is not a new benchmark run. The historical
`artifacts/e5_hyde_rerank/validation/weight_selection.json` is kept unchanged
as a hash/provenance reference and is no longer the source of applied weights.

Run outputs and executed notebooks are ignored local research artifacts. Keep
the task-specific artifact and cache directories separate from the historical
study directory, and use a new `--output-dir` for every notebook execution.
Each comparison run and selection manifest is written under a unique run ID.

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
python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_rerank_experiment.ipynb --output e5_rerank_smoke.executed.ipynb --output-dir artifacts/e5_rerank
```

Run the complete comparison with:

```bash
E5_RERANK_MODE=benchmark E5_RERANK_E5_BATCH_SIZE=128 E5_RERANK_BATCH_SIZE=128 E5_RERANK_TORCH_THREADS=8 python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_rerank_experiment.ipynb --output e5_rerank_full.executed.ipynb --output-dir artifacts/e5_rerank
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
E5_HYDE_MODE=smoke python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_hyde_experiment.ipynb --output e5_hyde_smoke.executed.ipynb --output-dir artifacts/e5_hyde
```

Run the complete declared benchmark only when model and dataset downloads and
available memory are sufficient:

```bash
cd code-retrieval
E5_HYDE_MODE=benchmark E5_HYDE_BATCH_SIZE=128 E5_HYDE_TORCH_THREADS=8 python -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=code-retrieval --ExecutePreprocessor.timeout=-1 notebooks/e5_hyde_experiment.ipynb --output e5_hyde_full.executed.ipynb --output-dir artifacts/e5_hyde
```

HyDE output is written below `code-retrieval/artifacts/e5_hyde/`. Generated
notebooks, caches, results, and metadata are ignored by Git; use a unique
`--output-dir` for each run when preserving executed notebooks.

The historical selection record at
`artifacts/e5_hyde_rerank/validation/weight_selection.json` remains preserved.
The combined notebook now recomputes all validation sweeps and applies weights
only from the emitted, hash-verified manifest; it does not read the historical
fixed-weight record as the decision source. Never select weights from test
results.

The official COIR evaluator receives unique ordinal scores that preserve each
artifact's descending score order and saved insertion order for ties. This
avoids pytrec_eval treating equal-score documents as unordered ties in one
system while RRF turns those same ranks into unique scores in another.

The notebook's setup cell checks the pinned Python version and all packages in
`requirements.txt` before loading models.

## Executed notebooks and result artifacts

Use a new timestamped `--output-dir` for each direct `nbconvert` execution to
preserve its executed notebook with captured cell outputs. Each notebook writes
`result.json`, `metadata.json`, and its comparison output to the configured
artifact directory; model inputs and rankings are cached under the recorded
cache identity. These outputs are local research artifacts and are ignored by
Git.

Direct `nbconvert` does not create runner traces or per-query analysis exports.
Preserve those separately when the analysis workflow requires them. The
published CoIR E5-base/CosQA value of `0.3259` is an external paper reference,
not a same-run baseline-parity artifact.
