# Indonesian ASTE reproduction results

Status: **independent replication**. Original per-seed artifacts and the full seed list were not recovered.

**Chapter 5 target outcome: NOT MET (0/4 baseline means).**

The Chapter 5 draft describes selecting `Max-Test-F1` on the test data during training. This replication selects checkpoints on validation and evaluates the held-out test once, as required by the approved protocol. The original seed list, checkpoints, and run outputs were not recovered, so this is not a method-matched rerun. The primary measure retains all 418 implicit-aspect HoASA test triplets, which the current PyABSA EMCGCN adapter cannot predict; explicit-only scores below are secondary diagnostics and do not replace the primary outcome. These differences affect comparability but do not by themselves establish the full cause of any shortfall.

Scores are exact test triplet micro-F1 percentages. Only run manifests marked `complete` contribute; smoke, partial, blocked, failed, interrupted, or absent runs never become scores. Attempt counts for the HoASA EMCGCN baseline are summarized below.

Declared seeds: `0, 1, 2, 3, 52`. Expected runs per cell: 5.

| Train → test | Architecture | Encoder | State | n | Seed F1 values | Mean | SD | Median | Min | Max | Chapter 5 target | Δ | Target status |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hoasa | emcgcn | mbert | complete | 5/5 | 0:70.15, 1:71.26, 2:69.63, 3:71.73, 52:71.52 | 70.86 | 0.92 | 71.26 | 69.63 | 71.73 | 75.72 | -4.86 | below |
| hoasa | emcgcn | indobert | complete | 5/5 | 0:71.92, 1:72.10, 2:71.50, 3:72.20, 52:71.58 | 71.86 | 0.31 | 71.92 | 71.50 | 72.20 | 78.01 | -6.15 | below |
| hoasa | emcgcn | xlmr | complete | 5/5 | 0:71.63, 1:73.05, 2:72.78, 3:73.40, 52:73.12 | 72.80 | 0.69 | 73.05 | 71.63 | 73.40 | 77.97 | -5.17 | below |
| hoasa | emcgcn | deberta_absa | complete | 5/5 | 0:69.05, 1:70.77, 2:68.48, 3:69.51, 52:70.54 | 69.67 | 0.98 | 69.51 | 68.48 | 70.77 | 80.79 | -11.12 | below |
| hoasa | ssgcn | mbert | smoke only; failed attempts | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | — | — | not applicable |
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

| Encoder | Chapter 5 target | Primary full-gold mean | Primary Δ | Explicit-only mean (diagnostic) | Diagnostic Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| mbert | 75.72 | 70.86 | -4.86 | 77.68 | 1.96 |
| indobert | 78.01 | 71.86 | -6.15 | 78.80 | 0.79 |
| xlmr | 77.97 | 72.80 | -5.17 | 79.91 | 1.94 |
| deberta_absa | 80.79 | 69.67 | -11.12 | 76.17 | -4.62 |

## HoASA EMCGCN attempt history

Complete seed runs contribute to the score table. Smoke, failed, and interrupted attempts remain in the run archive and are excluded from benchmark metrics.

| Encoder | Complete seed runs | Smoke attempts | Failed attempts | Interrupted attempts |
| --- | ---: | ---: | ---: | ---: |
| mbert | 5/5 | 3 | 6 | 3 |
| indobert | 5/5 | 0 | 0 | 0 |
| xlmr | 5/5 | 0 | 1 | 0 |
| deberta_absa | 5/5 | 0 | 2 | 0 |

## Secondary metrics

Each value is mean precision/recall/F1 (%) over complete seeds only. Explicit-only triplet scores exclude implicit-aspect gold; primary triplet scores above retain it.

| Train → test | Architecture | Encoder | Explicit-only triplet P/R/F1 | Aspect P/R/F1 | Opinion P/R/F1 | POS P/R/F1 | NEU P/R/F1 | NEG P/R/F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hoasa | emcgcn | mbert | 82.31/73.66/77.68 | 91.24/79.45/84.91 | 91.63/68.25/78.20 | 86.08/62.77/72.52 | 42.00/6.67/10.11 | 77.49/62.06/68.88 |
| hoasa | emcgcn | indobert | 83.82/74.45/78.80 | 91.73/79.56/85.19 | 93.05/68.66/78.98 | 86.38/63.56/73.18 | 41.82/6.67/10.00 | 80.58/62.55/70.35 |
| hoasa | emcgcn | xlmr | 86.02/74.62/79.91 | 93.24/79.32/85.72 | 94.42/67.98/79.04 | 89.36/63.53/74.26 | 26.67/4.44/7.33 | 81.60/62.97/71.06 |
| hoasa | emcgcn | deberta_absa | 77.97/74.57/76.17 | 88.80/80.38/84.36 | 89.31/69.64/78.24 | 82.12/63.45/71.52 | 0.00/0.00/0.00 | 72.72/63.02/67.46 |
| hoasa | ssgcn | mbert | — | — | — | — | — | — |
| hoasa | ssgcn | indobert | — | — | — | — | — | — |
| hoasa | ssgcn | xlmr | — | — | — | — | — | — |
| hoasa | ssgcn | deberta_absa | — | — | — | — | — | — |
| casa | emcgcn | mbert | — | — | — | — | — | — |
| casa | emcgcn | indobert | — | — | — | — | — | — |
| casa | emcgcn | xlmr | — | — | — | — | — | — |
| casa | emcgcn | deberta_absa | — | — | — | — | — | — |
| casa | ssgcn | mbert | — | — | — | — | — | — |
| casa | ssgcn | indobert | — | — | — | — | — | — |
| casa | ssgcn | xlmr | — | — | — | — | — | — |
| casa | ssgcn | deberta_absa | — | — | — | — | — | — |
| hoasa → casa | emcgcn | deberta_absa | — | — | — | — | — | — |
| hoasa → casa | ssgcn | deberta_absa | — | — | — | — | — | — |

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
