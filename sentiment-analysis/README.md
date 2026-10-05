# Indonesian ASTE reproduction pipeline

This project provides pinned source audits, HoASA adapters, PyABSA-compatible
EMCGCN and SSGCN training, leakage-controlled evaluation, and a seed-aware
report for the approved independent replication. It does not treat smoke runs
as benchmark results.

## Setup

From the workspace root, install the locked Python 3.10 environment:

```bash
uv sync --locked --python 3.10 --project sentiment-analysis
```

PyABSA 2.4.3 requires Python 3.10 and Transformers below 4.30. The lock pins a
CPU PyTorch build, compatible `update-checker` API, and protobuf 3.20.3 for the
Transformers 4.29 DeBERTa-v2 tokenizer bindings. Parser/model weights are
downloaded to ignored local data/cache directories and their revisions or file
hashes are recorded in each run.

The thesis's 128-token cap cannot preserve all HoASA train/dev sentences with
every pinned encoder. The runner uses the smallest per-encoder cap from the
frozen train/dev length audit (128–208 tokens) and verifies the source split
hashes before training. The test set does not determine this setting. PyABSA's
pairwise graph refinement uses a memory-efficient decomposition of the same
linear projection, keeping its weights and batch size unchanged.

PyABSA 2.4.3 assigns a separate hard-coded `1e-3` learning rate to EMCGCN's
graph and classifier parameters, even when `config.learning_rate` is set. The
runner overrides every optimizer group's rate to the matrix value (`2e-5`).
Run manifests identify this protocol; earlier runs using the framework default
remain archived but are excluded from its aggregates.

## Source and readiness audits

```bash
uv run --python 3.10 --project sentiment-analysis aste-audit --fetch --write sentiment-analysis/results/audits/source-audit-2026-09-29.json
uv run --python 3.10 --project sentiment-analysis aste-audit --require-casa-gold
```

The CASA command returns status 2 until complete, human-adjudicated,
sentiment-bearing ASTE gold, an approved aspect-to-category mapping, and a
duplicate-row policy are available. Do not bypass that gate.

## Parser and model runs

```bash
uv run --python 3.10 --project sentiment-analysis aste-download-parser
uv run --python 3.10 --project sentiment-analysis aste-run --dataset hoasa --architecture emcgcn --encoder mbert --seed 52 --purpose smoke --smoke-examples 24
```

Change `--purpose smoke` to `benchmark` only for a full declared split and
seed. Each run receives a unique directory under `sentiment-analysis/runs/`.
The true HoASA test split remains outside PyABSA until validation selects the
checkpoint, then is evaluated once. Implicit aspects remain in primary gold;
the explicit-model view excludes them and records that count. Smoke outputs are
never promoted to benchmark scores.

The matrix declares EMCGCN and the paper-informed SSGCN extension with mBERT,
IndoBERT, XLM-RoBERTa, and DeBERTa-ABSA. CASA and HoASA-to-CASA transfer remain
blocked until their sentiment-bearing human ASTE gold is frozen.

## Research notebook

Use [`notebooks/aste-reproduction-experiments.ipynb`](notebooks/aste-reproduction-experiments.ipynb) to inspect live and completed runs, launch an explicitly enabled benchmark condition or continue a missing HoASA seed matrix, summarize measured seed scores, and refresh the aggregate report. It delegates training to `aste-run`; execution is disabled by default and overlapping benchmark queues are rejected.

## Report

```bash
uv run --python 3.10 --project sentiment-analysis aste-report --output sentiment-analysis/results/aste-reproduction-report.md
```

Only `complete` run manifests contribute to benchmark scores. The report
includes all declared cells and seeds, Chapter 5 target comparisons, paired
architecture deltas, explicit-only diagnostics, term metrics, and per-polarity
metrics. Missing or blocked results remain visibly unmeasured.
