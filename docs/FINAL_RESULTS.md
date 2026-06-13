# Final Frozen MovieLens Ranking Results

Generated from the committed benchmark artifacts on 2026-06-13.

## Decision

- Promoted champion: `combined_feature_only`
- Canonical promotion baseline: `core_champion_replay`
- Champion NDCG@10: `0.322034`
- Absolute NDCG@10 improvement: `+0.009435`
- Relative NDCG@10 improvement: `+3.02%`
- Promotion eligibility: `true`
- Launch decision: `ITERATE`

The promoted model is the best eligible run under the current frozen-contract comparison. The higher legacy `core_compact` score is retained only as profile-selection context and is not treated as a directly comparable promotion candidate.

## Canonical comparison

| Metric | Canonical baseline | Promoted champion | Relative change |
|---|---:|---:|---:|
| NDCG@10 | 0.312599 | 0.322034 | +3.02% |
| MRR@10 | 0.509456 | 0.546196 | +7.21% |
| Recall efficiency@10 | 0.267388 | 0.265705 | -0.63% |
| Hit rate@10 | 0.636364 | 0.666667 | +4.76% |
| Genre/tag NDCG | 0.484022 | 0.498418 | +2.97% |
| Mood/decade NDCG | 0.095405 | 0.102074 | +6.99% |
| Personalized NDCG | 0.217080 | 0.216694 | -0.18% |
| Similar-to NDCG | 0.096545 | 0.124186 | +28.63% |
| Visual-query NDCG | 0.709141 | 0.706743 | -0.34% |
| ECE | 0.004906 | 0.005177 | +5.51% |
| p95 latency (ms) | 330.873465 | 198.038430 | -40.15% |

## Contract interpretation

The canonical manifest and query-group split were validated across:

- `core_champion_replay`
- `tag_genome_feature_only`
- `imdb_feature_only`
- `combined_feature_only`

Legacy profile runs are preserved for historical context. They do not support a strict GPU replay claim because their configuration fingerprints and, in one case, split contract differ from the canonical replay.

## Promotion rule

`same frozen manifest/splits, no retrieval-text drift, positive ranker lift, no >10% slice regression, non-negative overall NDCG change`

## Source artifacts

- `artifacts/frozen_quality_ablation/frozen_ablation_results.csv`
- `artifacts/frozen_quality_ablation/frozen_ablation_summary.json`
- `artifacts/frozen_quality_ablation/frozen_replay_diagnostics.json`
- `reports/frozen_quality_ablation/29_frozen_movielens_ablation.md`
