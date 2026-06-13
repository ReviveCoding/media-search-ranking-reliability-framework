# Dataset-Only Local and GitHub Runnability Verification

This audit validates the public-data execution path using two small, faithful MovieLens layouts:

- `dat`: MovieLens 1M-style `movies.dat`, `ratings.dat`, and `users.dat`, intentionally without tags.
- `csv`: modern MovieLens-style `movies.csv`, `ratings.csv`, and `tags.csv`.

Overall decision: **PASS**

| Format | Audit | Launch output | Ranker backend | NDCG@10 | Ranker lift vs hybrid |
|---|---:|---:|---|---:|---:|
| dat | PASS | REVIEW | lightgbm-lambdarank-cpu | 0.3292 | +0.1837 |
| csv | PASS | REVIEW | lightgbm-lambdarank-cpu | 0.3292 | +0.1837 |

## Interpretation

PASS means both MovieLens file-layout paths completed ingestion, ranking, evaluation, and artifact generation. A launch decision of REVIEW or ITERATE is a data-quality/model-quality outcome, not a runnability failure.

The fixture data is generated only to exercise the exact public-data loaders and end-to-end path. It is not a substitute for final MovieLens benchmark metrics.
