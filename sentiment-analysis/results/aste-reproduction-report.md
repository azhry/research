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

The native-rate history contains 20 complete measured runs (five per encoder) but predates explicit protocol IDs; its manifests record the expected group rates and all runs remain under their original identities. Those full-gold means are below all four Chapter 5 references. The second schedule is the completed uniform-rate protocol, now superseded because its adapter overrode PyABSA's graph/classifier rates. Neither historical schedule is relabeled as an active-protocol run. The active v3 protocol has 0/20 complete HoASA EMCGCN seed runs; those measurements are reported separately below and are not pooled with historical schedules.

The explicit-only scores below use the same predictions and exclude implicit-aspect triplets from the gold denominator. Chapter 5 does not document its implicit-label policy, and it selects the best test-set epoch while this replication selects on validation. These figures are sensitivity diagnostics, not evidence that a target has been reproduced.

| Schedule | Encoder | Runs | Full-gold seed F1 (%) | Full-gold mean | Explicit-only seed F1 (%) | Explicit-only mean | Chapter 5 reference | Full-gold Δ | Explicit-only Δ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Native PyABSA groups | mbert | 0 | — | — | — | — | 75.72 | — | — |
| Native PyABSA groups | indobert | 0 | — | — | — | — | 78.01 | — | — |
| Native PyABSA groups | xlmr | 0 | — | — | — | — | 77.97 | — | — |
| Native PyABSA groups | deberta_absa | 0 | — | — | — | — | 80.79 | — | — |
| Superseded uniform groups | mbert | 0 | — | — | — | — | 75.72 | — | — |
| Superseded uniform groups | indobert | 0 | — | — | — | — | 78.01 | — | — |
| Superseded uniform groups | xlmr | 0 | — | — | — | — | 77.97 | — | — |
| Superseded uniform groups | deberta_absa | 0 | — | — | — | — | 80.79 | — | — |

| Train → test | Architecture | Encoder | State | n | Seed F1 values | Mean | SD | Median | Min | Max | Chapter 5 target | Δ | Target status |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hoasa | emcgcn | mbert | smoke only | 0/5 | 0:—, 1:—, 2:—, 3:—, 52:— | — | — | — | — | — | 75.72 | — | not measured |
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

These rows summarize the active native-group protocol. The primary full-gold metric retains implicit-aspect triplets; explicit-only diagnostics do not establish a Chapter 5 match because its implicit-label policy is undocumented.

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
| mbert | 0/5 | 2 | 0 | 0 |
| indobert | 0/5 | 0 | 0 | 0 |
| xlmr | 0/5 | 0 | 0 | 0 |
| deberta_absa | 0/5 | 0 | 0 | 0 |

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

## Historical PyABSA source-script reference (v1; test-selected)

This immutable v1 run set predates the dataset-cache seed-restoration fix. It remains historical evidence and is not pooled with the v2 full rerun.
This reproduction runs the four unchanged local PyABSA source scripts, one per encoder, with each script's declared seeds `1, 2, 3`. It reports `MetricVisualizer Max-Test-F1`, which selects the best checkpoint using test-set scores during training. These measurements are test-leaked reference diagnostics and are not clean held-out estimates; they are never pooled with the validation-selected v3 results above.

Every complete row below is recomputed from the three per-seed metrics in its source-run manifest. The source script, resolved configuration, cached model revision, train/dev/test data copies, and every saved run artifact checksum are validated before the row is emitted. Workspace source/config files are snapshotted under each run's `provenance/workspace-source/` so later code edits do not invalidate historical provenance.
PyABSA 2.4.3 treats source-config `log_step=-1` as a runtime sentinel and replaces it with the training dataloader length. The runner records the installed instructor source checksum, snapshots that source, and verifies the three seed logs (`-1` on initialization, then the resolved value) and `Num steps` before accepting this normalization. Any earlier validation rejection corrected under this rule retains the byte-for-byte original failed manifest and a checksum-linked correction record; source scripts and training settings remain unchanged.

| Encoder | State | Outcomes | Seed Max-Test-F1 (%) | Mean (%) | Reported reference mean (%) | Reference n | Δ vs reference |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| indobert | failed | 0/3; attempts failed=1 | 1:—, 2:—, 3:— | — | 78.01 | 2 | — |
| mbert | not run | 0/3 | 1:—, 2:—, 3:— | — | 75.72 | 3 | — |
| xlmr | not run | 0/3 | 1:—, 2:—, 3:— | — | 77.97 | 3 | — |
| deberta_absa | failed | 0/3; attempts failed=1 | 1:—, 2:—, 3:— | — | 80.79 | 3 | — |
Source-script attempts snapshot the installed PyABSA cache-loader source and check whether a dataset-cache hit replaced the requested seed. Any such restoration is recorded and verified before scores are reported.


Reported targets are the local PyABSA log means frozen in `configs/experiment-matrix.json`. The IndoBERT reference had two recovered scores with unknown seed identities (`n=2`); the three declared source-script seeds remain explicit and are not represented as a matched rerun. A below-target result is reported as measured and is not retried or retuned based on test scores.


## PyABSA source-script full rerun (v2; test-selected)

This immutable v2 run set repeats all four unchanged local source scripts and all three declared seeds after the runtime cache-seed correction. The correction changes only the requested seed restored after a cache hit; source scripts and training settings remain unchanged.
This reproduction runs the four unchanged local PyABSA source scripts, one per encoder, with each script's declared seeds `1, 2, 3`. It reports `MetricVisualizer Max-Test-F1`, which selects the best checkpoint using test-set scores during training. These measurements are test-leaked reference diagnostics and are not clean held-out estimates; they are never pooled with the validation-selected v3 results above.

Every complete row below is recomputed from the three per-seed metrics in its source-run manifest. The source script, resolved configuration, cached model revision, train/dev/test data copies, and every saved run artifact checksum are validated before the row is emitted. Workspace source/config files are snapshotted under each run's `provenance/workspace-source/` so later code edits do not invalidate historical provenance.
PyABSA 2.4.3 treats source-config `log_step=-1` as a runtime sentinel and replaces it with the training dataloader length. The runner records the installed instructor source checksum, snapshots that source, and verifies the three seed logs (`-1` on initialization, then the resolved value) and `Num steps` before accepting this normalization. Any earlier validation rejection corrected under this rule retains the byte-for-byte original failed manifest and a checksum-linked correction record; source scripts and training settings remain unchanged.

| Encoder | State | Outcomes | Seed Max-Test-F1 (%) | Mean (%) | Reported reference mean (%) | Reference n | Δ vs reference |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| indobert | complete; test-leaked | 3/3; attempts failed=2 | 1:75.61, 2:76.05, 3:80.41 | 77.36 | 78.01 | 2 | -0.65 |
| mbert | complete; test-leaked | 3/3; attempts failed=1, interrupted=1 | 1:75.03, 2:76.12, 3:76.01 | 75.72 | 75.72 | 3 | -0.00 |
| xlmr | complete; test-leaked | 3/3; attempts interrupted=1 | 1:78.78, 2:77.17, 3:77.96 | 77.97 | 77.97 | 3 | +0.00 |
| deberta_absa | complete; test-leaked | 3/3; attempts failed=1, interrupted=1 | 1:76.26, 2:75.11, 3:82.26 | 77.88 | 80.79 | 3 | -2.91 |
For source-script attempts with dataset-cache hits, the PyABSA 2.4.3 cache key excludes `seed` while the cached object includes a serialized `ConfigManager`. The runner preserves the requested seed across cache restores, snapshots and checksums the installed cache-loader source, and records each correction. Recorded events: requested seed 3 was restored after a cached config supplied seed 2, requested seed 3 was restored after a cached config supplied seed 2, requested seed 3 was restored after a cached config supplied seed 2, requested seed 3 was restored after a cached config supplied seed 2. The source scripts and declared training settings are unchanged.


Reported targets are the local PyABSA log means frozen in `configs/experiment-matrix.json`. The IndoBERT reference had two recovered scores with unknown seed identities (`n=2`); the three declared source-script seeds remain explicit and are not represented as a matched rerun. A below-target result is reported as measured and is not retried or retuned based on test scores.
