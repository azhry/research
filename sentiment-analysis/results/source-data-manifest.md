# Sentiment Analysis Source Data Manifest

Audited on 2026-09-29. Hashes are SHA-256 of the exact upstream bytes at the
immutable commit shown. Review text is intentionally omitted from this report.

## HoASA ASTE source

Repository: [`rdyzakya/IndoLEGO-ABSA`](https://github.com/rdyzakya/IndoLEGO-ABSA)

Revision: `0bb3880fed6caf28e05b4eccf329b642e064fd0a`

| Split | Path | Sentences | Triplets | Implicit-aspect triplets (`aspect == [-1]`) | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Train | `data/train.txt` | 3,000 | 7,551 | 1,294 | `33714cfdafa40a5c581cb744845268aeea18450265af67ec572623afa7c20b6d` |
| Dev | `data/dev.txt` | 1,000 | 2,458 | 469 | `a00d16dcb0bcbc6cde1c9650323145c5d22be33620eff7c812fafe36ec06c690` |
| Test | `data/test.txt` | 1,000 | 2,709 | 418 | `3f3883f8c25706d79b1c6bbce253bb548b38c4eb4a786f4b14506d3a28a749a0` |

The Chapter 3 train/test sentence and triplet totals match these files. The
source split is now recoverable. Chapter 6's different split and the Chapter 5
training artifacts, seed list, checkpoint-selection protocol, and per-seed
scores remain unverified. New runs must be called independent replications.

## CASA category-polarity source

Repository: [`indobenchmark/indonlu`](https://github.com/indobenchmark/indonlu)

Revision: `ce728f6926a36174b9923dfe49d6a6839b6e9bb7`

| Split | Path | Rows | Labels | SHA-256 |
| --- | --- | ---: | --- | --- |
| Train | `dataset/casa_absa-prosa/train_preprocess.csv` | 810 | fuel, machine, others, part, price, service | `ffd2a88edf5e270cea79ad84d2ca4170c9a2fd71a38540280d5eb3b95d261f76` |
| Validation | `dataset/casa_absa-prosa/valid_preprocess.csv` | 90 | fuel, machine, others, part, price, service | `4ea114d060796e59944b1cf7f0ad7950bd0532024348a17d0f7c6b6464328424` |
| Test | `dataset/casa_absa-prosa/test_preprocess.csv` | 180 | fuel, machine, others, part, price, service | `843564f24c10d8360fe63395a821f94eb46985abddb94dba6865fcd0a29c70c7` |

The published rows align one-to-one by exact review text with the three local
Label Studio exports. There is one duplicate review in train; both source rows
have the same category-polarity vector. The public files provide category
sentiment only, not aspect/opinion terms or their links.

## Existing local CASA ASTE annotations

These user-workspace files are not complete gold sets yet. Their current SHA-256
and annotation coverage are:

| Split | File | Tasks | Tasks with any span and relation annotations | Relations | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Train | `sentiment-analysis/docs/thesis/data/labelstudio_import/casa_train.json` | 810 | 436 | 970 | `a359a9d3fa1a5b3a6e05e35564587fdb1da366308c5412263845c392e5f85c30` |
| Validation | `sentiment-analysis/docs/thesis/data/labelstudio_import/casa_valid.json` | 90 | 48 | 92 | `db6754a729d9e247209ac3a82d9314c1de9a184fd2cd5ea2bd13a919c5d410e8` |
| Test | `sentiment-analysis/docs/thesis/data/labelstudio_import/casa_test.json` | 180 | 93 | 170 | `712d2de9fafe41a64e444f7f25c8531b9c2a1ad849143e55adf90d940ed0d611` |

The remaining tasks have no span annotations in the audited exports. None of
the exported aspect-opinion relations carries a sentiment label. The human
annotation coverage must be completed, category mapping adjudicated, and the
final ASTE splits frozen before any CASA model training or test evaluation.
Do not promote automatic/LLM-generated labels to gold.

## Local placeholders

The files under `sentiment-analysis/docs/thesis/data/casa_csv/` currently hold
only `404: Not Found` and must never be read as dataset rows.
