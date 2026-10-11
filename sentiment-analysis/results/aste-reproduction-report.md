# Indonesian ASTE reproduction results

Status: **independent replication**. Original per-seed artifacts and the full seed list were not recovered.

Active run protocol: `indonesian-aste-matrix-v3-pyabsa-native-groups`. Set config.learning_rate to 2e-5 and preserve PyABSA 2.4.3 EMCGCN native groups: 2e-5 for Transformer parameters and 1e-3 for graph/classifier parameters.

**Chapter 5 target outcome: NOT YET FULLY MEASURED.**

The Chapter 5 draft describes selecting `Max-Test-F1` on the test data during training. This replication selects checkpoints on validation and evaluates the held-out test once, as required by the approved protocol. The original seed list, checkpoints, and run outputs were not recovered, so this is not a method-matched rerun. The primary measure retains all 418 implicit-aspect HoASA test triplets, which the current PyABSA EMCGCN adapter cannot predict; explicit-only scores below are secondary diagnostics and do not replace the primary outcome.

The active native-group protocol has no complete runs yet, so its implicit-aspect sensitivity is not measured. Completed native-group and superseded uniform-rate runs are summarized separately below.

## Optimizer schedule correction

Source inspection of pinned PyABSA 2.4.3 shows that `config.learning_rate=2e-5` is used for the two Transformer parameter groups; its EMCGCN instructor separately assigns `1e-3` to both graph/classifier groups. A superseded adapter incorrectly forced all four groups to `2e-5`. In matched mBERT seed-0 runs with the same data hashes, model revision, seed, and sequence length, native rates measured 70.15 full-gold F1 / 76.76 explicit-only F1, while the uniform-rate run measured 67.51 / 73.69. The native schedule is restored in the active code and configuration. Across the five-seed native-rate history, all four full-gold means remain below the Chapter 5 references; only three of four explicit-only means meet them, and that secondary metric does not establish exact reproduction. Completed uniform-rate runs remain historical evidence under their original protocol and are not pooled with the active protocol.

Scores are exact test triplet micro-F1 percentages. Only run manifests marked `complete` contribute; smoke, partial, blocked, failed, interrupted, or absent runs never become scores. Attempt counts for the HoASA EMCGCN baseline are summarized below.

Declared seeds: `0, 1, 2, 3, 52`. Expected runs per cell: 5.

## Completed historical optimizer schedules

The native-rate history contains 20 complete measured runs (five per encoder) but predates explicit protocol IDs; its manifests record the expected group rates and all runs remain under their original identities. Those full-gold means are below all four Chapter 5 references. The second schedule is the completed uniform-rate protocol, now superseded because its adapter overrode PyABSA's graph/classifier rates. Neither historical schedule is relabeled as an active-protocol run. No duplicate 20-run queue was started during this correction; the active v3 rows remain `not run` until runs with that protocol ID are deliberately collected.

The explicit-only scores below use the same predictions and exclude implicit-aspect triplets from the gold denominator. Chapter 5 does not document its implicit-label policy, and it selects the best test-set epoch while this replication selects on validation. These figures are sensitivity diagnostics, not evidence that a target has been reproduced.

| Schedule | Encoder | Runs | Full-gold seed F1 (%) | Full-gold mean | Explicit-only seed F1 (%) | Explicit-only mean | Chapter 5 reference | Full-gold Δ | Explicit-only Δ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Native PyABSA groups | mbert | 5 | 0:70.15, 1:71.26, 2:69.63, 3:71.73, 52:71.52 | 70.86 | 0:76.76, 1:78.21, 2:76.07, 3:78.88, 52:78.49 | 77.68 | 75.72 | -4.86 | +1.96 |
| Native PyABSA groups | indobert | 5 | 0:71.92, 1:72.10, 2:71.50, 3:72.20, 52:71.58 | 71.86 | 0:78.68, 1:78.93, 2:78.52, 3:79.14, 52:78.75 | 78.80 | 78.01 | -6.15 | +0.79 |
| Native PyABSA groups | xlmr | 5 | 0:71.63, 1:73.05, 2:72.78, 3:73.40, 52:73.12 | 72.80 | 0:78.52, 1:80.16, 2:79.97, 3:80.58, 52:80.31 | 79.91 | 77.97 | -5.17 | +1.94 |
| Native PyABSA groups | deberta_absa | 5 | 0:69.05, 1:70.77, 2:68.48, 3:69.51, 52:70.54 | 69.67 | 0:75.30, 1:77.47, 2:74.61, 3:76.11, 52:77.35 | 76.17 | 80.79 | -11.12 | -4.62 |
| Superseded uniform groups | mbert | 5 | 0:67.51, 1:66.95, 2:67.16, 3:69.95, 52:69.25 | 68.16 | 0:73.69, 1:73.16, 2:73.70, 3:76.73, 52:75.86 | 74.63 | 75.72 | -7.56 | -1.09 |
| Superseded uniform groups | indobert | 5 | 0:71.36, 1:71.53, 2:71.01, 3:72.25, 52:70.77 | 71.38 | 0:78.36, 1:78.29, 2:77.66, 3:79.11, 52:77.68 | 78.22 | 78.01 | -6.63 | +0.21 |
| Superseded uniform groups | xlmr | 5 | 0:71.55, 1:69.48, 2:70.56, 3:69.67, 52:69.53 | 70.16 | 0:78.30, 1:75.82, 2:77.25, 3:75.98, 52:76.33 | 76.74 | 77.97 | -7.81 | -1.23 |
| Superseded uniform groups | deberta_absa | 5 | 0:66.11, 1:65.04, 2:68.83, 3:64.31, 52:64.56 | 65.77 | 0:72.09, 1:71.20, 2:75.47, 3:70.02, 52:71.19 | 71.99 | 80.79 | -15.02 | -8.80 |

| Train → test | Architecture | Encoder | State | n | Seed F1 values | Mean | SD | Median | Min | Max | Chapter 5 target | Δ | Target status |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hoasa | emcgcn | mbert | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 75.72 | — | not measured |
| hoasa | emcgcn | indobert | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 78.01 | — | not measured |
| hoasa | emcgcn | xlmr | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 77.97 | — | not measured |
| hoasa | emcgcn | deberta_absa | not run | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 80.79 | — | not measured |
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

These rows refer only to the active native-group protocol. It has no complete runs yet; completed measurements under the historical schedules are shown above. The primary full-gold metric retains implicit-aspect triplets; explicit-only diagnostics do not establish that a Chapter 5 target was reproduced.

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
| mbert | 0/5 | 0 | 0 | 0 |
| indobert | 0/5 | 0 | 0 | 0 |
| xlmr | 0/5 | 0 | 0 | 0 |
| deberta_absa | 0/5 | 0 | 0 | 0 |

## Superseded uniform-rate attempt history

The superseded uniform-rate protocol queue ended with all 20 declared seed conditions complete and no queue-level failures. Earlier failed and interrupted attempts remain in the run archive and are not scores.

| Encoder | Complete runs | Failed attempts | Interrupted attempts |
| --- | ---: | ---: | ---: |
| mbert | 5/5 | 5 | 1 |
| indobert | 5/5 | 5 | 0 |
| xlmr | 5/5 | 5 | 1 |
| deberta_absa | 5/5 | 5 | 1 |

## Historical native-rate attempt history

These unprotocolized manifests record the native group rates. Only complete runs with metrics contribute to the historical score table; smoke, failed, and interrupted attempts remain excluded.

| Encoder | Complete runs | Failed attempts | Interrupted attempts | Smoke attempts |
| --- | ---: | ---: | ---: | ---: |
| mbert | 5/5 | 0 | 0 | 3 |
| indobert | 5/5 | 0 | 0 | 0 |
| xlmr | 5/5 | 0 | 0 | 0 |
| deberta_absa | 5/5 | 0 | 0 | 0 |

## Unprotocolized attempts without optimizer metadata

These older failed, interrupted, or smoke manifests do not record resolved optimizer groups, so they cannot be assigned to either schedule and contribute no scores. They remain in the run archive.

| Encoder | Failed | Interrupted | Smoke |
| --- | ---: | ---: | ---: |
| mbert | 6 | 3 | 0 |
| indobert | 0 | 0 | 0 |
| xlmr | 1 | 0 | 0 |
| deberta_absa | 2 | 0 | 0 |

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
