# Indonesian ASTE reproduction results

Status: **independent replication**. Original per-seed artifacts and the full seed list were not recovered.

Active run protocol: `indonesian-aste-matrix-v2-uniform-learning-rate`. Every optimizer parameter group uses the configured learning rate of 2e-05.

**Chapter 5 target outcome: NOT YET FULLY MEASURED.**

The Chapter 5 draft describes selecting `Max-Test-F1` on the test data during training. This replication selects checkpoints on validation and evaluates the held-out test once, as required by the approved protocol. The original seed list, checkpoints, and run outputs were not recovered, so this is not a method-matched rerun. The primary measure retains all 418 implicit-aspect HoASA test triplets, which the current PyABSA EMCGCN adapter cannot predict; explicit-only scores below are secondary diagnostics and do not replace the primary outcome.

Explicit-only sensitivity is not yet measurable under the active uniform-rate protocol. Archived runs with PyABSA's default optimizer schedule are shown separately below and do not establish a match to the Chapter 5 targets.

Scores are exact test triplet micro-F1 percentages. Only run manifests marked `complete` contribute; smoke, partial, blocked, failed, interrupted, or absent runs never become scores. Attempt counts for the HoASA EMCGCN baseline are summarized below.

Declared seeds: `0, 1, 2, 3, 52`. Expected runs per cell: 5.

## Earlier complete runs with PyABSA's default optimizer groups

These complete runs used PyABSA 2.4.3's optimizer groups: 2e-5 for Transformer parameters and 1e-3 for graph/classifier parameters. The Chapter 5 configuration table lists `learning_rate=2e-5` but does not say whether the graph groups were overridden. The active protocol explicitly uses 2e-5 for every group, so the two schedules are reported separately and are not pooled. The earlier runs remain measured evidence under their recorded settings.

The explicit-only scores below use the same predictions and exclude implicit-aspect triplets from the gold denominator. Chapter 5 does not document its implicit-label policy, and it selects the best test-set epoch while this replication selects on validation. These figures are sensitivity diagnostics, not evidence that a target has been reproduced.

| Encoder | Runs | Full-gold seed F1 (%) | Full-gold mean | Explicit-only seed F1 (%) | Explicit-only mean | Chapter 5 reference | Diagnostic Δ vs reference |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| mbert | 5 | 0:70.15, 1:71.26, 2:69.63, 3:71.73, 52:71.52 | 70.86 | 0:76.76, 1:78.21, 2:76.07, 3:78.88, 52:78.49 | 77.68 | 75.72 | +1.96 |
| indobert | 5 | 0:71.92, 1:72.10, 2:71.50, 3:72.20, 52:71.58 | 71.86 | 0:78.68, 1:78.93, 2:78.52, 3:79.14, 52:78.75 | 78.80 | 78.01 | +0.79 |
| xlmr | 5 | 0:71.63, 1:73.05, 2:72.78, 3:73.40, 52:73.12 | 72.80 | 0:78.52, 1:80.16, 2:79.97, 3:80.58, 52:80.31 | 79.91 | 77.97 | +1.94 |
| deberta_absa | 5 | 0:69.05, 1:70.77, 2:68.48, 3:69.51, 52:70.54 | 69.67 | 0:75.30, 1:77.47, 2:74.61, 3:76.11, 52:77.35 | 76.17 | 80.79 | -4.62 |

| Train → test | Architecture | Encoder | State | n | Seed F1 values | Mean | SD | Median | Min | Max | Chapter 5 target | Δ | Target status |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hoasa | emcgcn | mbert | running; failed attempts | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 75.72 | — | not measured |
| hoasa | emcgcn | indobert | failed | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 78.01 | — | not measured |
| hoasa | emcgcn | xlmr | failed | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 77.97 | — | not measured |
| hoasa | emcgcn | deberta_absa | failed | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 80.79 | — | not measured |
| hoasa | ssgcn | mbert | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| hoasa | ssgcn | indobert | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| hoasa | ssgcn | xlmr | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| hoasa | ssgcn | deberta_absa | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | emcgcn | mbert | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | emcgcn | indobert | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | emcgcn | xlmr | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | emcgcn | deberta_absa | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | ssgcn | mbert | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | ssgcn | indobert | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | ssgcn | xlmr | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| casa | ssgcn | deberta_absa | blocked: human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| hoasa → casa | emcgcn | deberta_absa | blocked: CASA human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
| hoasa → casa | ssgcn | deberta_absa | blocked: CASA human gold | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |

## Chapter 5 reference diagnostics

The table separates the primary full-gold metric from explicit-only diagnostics. The latter exclude implicit-aspect triplets and are not used to declare the Chapter 5 target met.

| Encoder | Chapter 5 target | Primary full-gold mean | Primary Δ | Explicit-only mean (diagnostic) | Diagnostic Δ | F1 change when implicit gold is excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mbert | 75.72 | — | — | — | — | — |
| indobert | 78.01 | — | — | — | — | — |
| xlmr | 77.97 | — | — | — | — | — |
| deberta_absa | 80.79 | — | — | — | — | — |

## HoASA EMCGCN attempt history

Complete seed runs contribute to the score table. Smoke, failed, and interrupted attempts remain in the run archive and are excluded from benchmark metrics.

| Encoder | Complete seed runs | Smoke attempts | Failed attempts | Interrupted attempts |
| --- | ---: | ---: | ---: | ---: |
| mbert | 0/5 | 0 | 5 | 0 |
| indobert | 0/5 | 0 | 5 | 0 |
| xlmr | 0/5 | 0 | 5 | 0 |
| deberta_absa | 0/5 | 0 | 5 | 0 |

## Secondary metrics

Each value is mean precision/recall/F1 (%) over complete seeds only. The full-source opinion metric includes opinion spans linked to implicit-aspect gold even though those triplets are absent from the PyABSA training view; the adjacent explicit-triplet-only opinion metric removes those spans from its denominator.

| Train → test | Architecture | Encoder | Explicit-only triplet P/R/F1 | Aspect P/R/F1 | Opinion P/R/F1 (full source) | Opinion P/R/F1 (explicit triplets) | POS P/R/F1 | NEU P/R/F1 | NEG P/R/F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hoasa | emcgcn | mbert | — | — | — | — | — | — | — |
| hoasa | emcgcn | indobert | — | — | — | — | — | — | — |
| hoasa | emcgcn | xlmr | — | — | — | — | — | — | — |
| hoasa | emcgcn | deberta_absa | — | — | — | — | — | — | — |
| hoasa | ssgcn | mbert | — | — | — | — | — | — | — |
| hoasa | ssgcn | indobert | — | — | — | — | — | — | — |
| hoasa | ssgcn | xlmr | — | — | — | — | — | — | — |
| hoasa | ssgcn | deberta_absa | — | — | — | — | — | — | — |
| casa | emcgcn | mbert | — | — | — | — | — | — | — |
| casa | emcgcn | indobert | — | — | — | — | — | — | — |
| casa | emcgcn | xlmr | — | — | — | — | — | — | — |
| casa | emcgcn | deberta_absa | — | — | — | — | — | — | — |
| casa | ssgcn | mbert | — | — | — | — | — | — | — |
| casa | ssgcn | indobert | — | — | — | — | — | — | — |
| casa | ssgcn | xlmr | — | — | — | — | — | — | — |
| casa | ssgcn | deberta_absa | — | — | — | — | — | — | — |
| hoasa → casa | emcgcn | deberta_absa | — | — | — | — | — | — | — |
| hoasa → casa | ssgcn | deberta_absa | — | — | — | — | — | — | — |

## Protocol and evidence limits

- Checkpoints are selected on validation triplet F1; held-out test predictions are produced and evaluated once per model.
- HoASA source `[-1]` implicit aspects remain in canonical gold and the primary test denominator. The report must also show explicit-only diagnostics when the selected model cannot represent implicit aspects.
- CASA cells stay blocked until all splits have complete, human-adjudicated, sentiment-bearing ASTE annotations, an approved aspect-to-category mapping, and frozen hashes.
- Every score must resolve to one immutable run directory with model/data revisions, seed, configuration, environment, checkpoint, predictions, and digests.
- No seed-level results are present unless an immutable run manifest and metrics file exist.

## Paired architecture deltas

Delta is SSGCN minus EMCGCN in test triplet micro-F1 percentage points, paired by dataset, encoder, and seed. Only seeds with complete runs for both architectures are included.

| Train → test | Encoder | Paired seeds | Per-seed Δ | Mean Δ |
| --- | --- | --- | --- | ---: |
| hoasa | mbert | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| hoasa | indobert | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| hoasa | xlmr | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| hoasa | deberta_absa | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| casa | mbert | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| casa | indobert | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| casa | xlmr | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| casa | deberta_absa | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
| hoasa → casa | deberta_absa | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — |
