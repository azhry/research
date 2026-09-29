# Sentiment Analysis

AZH-521 reproduces ASTE experiments on HoASA and CASA, comparing EMCGCN with
SSGCN using mBERT, IndoBERT, XLM-RoBERTa, and DeBERTa-ABSA. The primary metric
is exact aspect-opinion-sentiment triplet F1.

The approved AZH-521 plan starts with four HoASA EMCGCN baseline conditions.
The runner and commands are documented in `sentiment-analysis/README.md`;
experiment outputs and audits live in `sentiment-analysis/results/`.

HoASA has pinned train/dev/test splits. CASA remains blocked until complete,
human-adjudicated sentiment-bearing ASTE annotations and an approved
aspect-category mapping are available. Keep smoke runs separate from benchmark
evidence and preserve each run's provenance.
