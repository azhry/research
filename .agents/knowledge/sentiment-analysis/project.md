# Sentiment Analysis Research

## Research area

This project studies sentiment-bearing aspect-based opinion extraction in
Indonesian text. The current ASTE work uses hotel and automotive review domains.
HoASA and CASA are the corpora under study; future corpora should be added only
when their annotations and evaluation protocol meet the project's evidence
requirements.

The research compares extraction architectures, including graph-based models
such as EMCGCN and SSGCN, with Indonesian and multilingual Transformer encoders.
The exact model and encoder combinations belong to each experiment's
specification rather than defining the whole knowledge area.

## Evaluation and evidence

- The primary ASTE measure is exact aspect-opinion-sentiment triplet F1, with
  precision and recall reported alongside it. A match requires both spans and
  sentiment polarity to agree.
- Use the validation split for checkpoint and configuration selection. Keep the
  test split held out for final evaluation, and report its use explicitly.
- Preserve dataset versions, split identities, model configuration, random
  seeds, code revision, checkpoints, and outputs for every run. Do not overwrite
  prior runs or present smoke/partial runs as complete benchmark evidence.
- Keep measured results separate from hypotheses and state limitations that
  affect interpretation. Do not use synthetic or model-generated annotations
  as human gold evidence.

## Dataset boundaries

Preserve each corpus's annotation semantics through preprocessing and scoring.
In particular, HoASA includes implicit aspects; retain them in the canonical
gold data and primary evaluation denominator, and label any explicit-span-only
analysis as secondary. Do not silently drop examples that the model cannot
represent.

Before using CASA for sentiment-bearing ASTE evaluation, require complete,
human-adjudicated sentiment annotations and an approved aspect-category
mapping. If these prerequisites are unmet, describe CASA as blocked and do not
report it as an evaluated benchmark.

## Workspace map

- `sentiment-analysis/README.md` describes the reusable experiment workflow.
- `sentiment-analysis/plan-and-specifications.md` holds the current study's
  approved protocol and scope.
- `sentiment-analysis/results/` contains run-linked result summaries; run
  directories retain detailed metrics and provenance.

AZH-521 is the current reproduction work item in this research area. Its ticket
and run-specific plan determine the active deliverables; this page records
cross-study context and should remain useful when that work item changes.
