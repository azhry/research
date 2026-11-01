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

PyABSA 2.4.3's default EMCGCN optimizer uses `2e-5` for Transformer parameters
and `1e-3` for graph and classifier parameters. The active matrix preserves
those native groups. A superseded uniform-rate protocol applied `2e-5` to every
group by mistake; its completed runs remain archived and are reported separately
from the active protocol. Run manifests record each resolved group rate.

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

## Optional Modal execution

Modal runs on a single T4 GPU. The image starts from the locked Python 3.10
environment, then replaces its CPU-only Torch wheel with pinned
`torch==2.5.1+cu121`; the local runner remains CPU-only. The worker fails if
CUDA is unavailable. The image uses the tracked runner, project metadata, and
matrix; it stages only the HoASA/CASA source files, required CASA Label Studio
exports, and pinned Stanza parser directory. Complete the source and parser
setup above before running remotely. Authentication stays in your local Modal
configuration. The persistent
`aste-research-runs` Volume stores each unique run; after completion, only the
new run directory is copied under `sentiment-analysis/runs/`. Modal compute
and persistent storage may incur charges.

Install the optional client and configure the local Modal profile:

```bash
uv sync --locked --python 3.10 --project sentiment-analysis --extra modal
uv run --python 3.10 --project sentiment-analysis --extra modal modal setup
```

Run a small remote smoke:

```bash
uv run --python 3.10 --project sentiment-analysis --extra modal modal run \
  sentiment-analysis/src/aste_repro/modal_runner.py \
  --dataset hoasa --architecture emcgcn --encoder mbert \
  --seed 52 --purpose smoke --smoke-examples 2
```

The Modal entrypoint accepts the existing `aste-run` arguments and defaults
to smoke to avoid starting a full benchmark accidentally. Set
`--purpose benchmark` only for an explicitly approved benchmark condition.
The command prints the unique run ID, smoke/benchmark status, and local
artifact path. Modal Volumes keep remote artifacts durable; the local copy
uses the standard manifest, metrics, predictions, logs, and checkpoint layout.
For large GPU matrices, add `--artifact-mode summary` to keep manifests,
metrics, predictions, and logs locally while retaining checkpoints on the
`aste-research-runs` Volume. The generated `modal-storage.json` records the
remote run prefix and checkpoint hashes. The default `full` mode still copies
the complete artifact directory locally.
See [Modal Images](https://modal.com/docs/guide/images) and
[Modal Volumes](https://modal.com/docs/guide/volumes) for the provider behavior.

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
