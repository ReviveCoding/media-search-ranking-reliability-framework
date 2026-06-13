# Frozen MovieLens Ranking Quality Ablation

Promotion-eligible variants use the same frozen catalog fingerprint, queries, judgments, and query-group splits.
Legacy core profile runs are retained for historical/profile-selection context only and are not treated as strictly comparable.
Tag Genome and IMDb operate in feature-only mode and do not rewrite retrieval text.

| run_name                | variant                 | ranker_profile   |   ndcg_at_10 |   mrr_at_10 |   recall_efficiency_at_10 |   hit_rate_at_10 |   ranker_lift_vs_hybrid |   genre_tag_ndcg |   mood_decade_ndcg |   personalized_ndcg |   similar_to_ndcg |   visual_query_ndcg |        ece |   p95_latency_ms | launch_decision   | tag_genome_applied   | imdb_applied   | retrieval_text_changed   | eligible_for_promotion   |
|:------------------------|:------------------------|:-----------------|-------------:|------------:|--------------------------:|-----------------:|------------------------:|-----------------:|-------------------:|--------------------:|------------------:|--------------------:|-----------:|-----------------:|:------------------|:---------------------|:---------------|:-------------------------|:-------------------------|
| core_compact            | core                    | compact          |     0.325471 |    0.552902 |                  0.268735 |         0.656566 |                0.207948 |         0.492995 |          0.109212  |            0.23541  |         0.123792  |            0.703298 | 0.00497949 |          179.194 | ITERATE           | False                | False          | False                    | False                    |
| combined_feature_only   | combined_feature_only   | compact          |     0.322034 |    0.546196 |                  0.265705 |         0.666667 |                0.204511 |         0.498418 |          0.102074  |            0.216694 |         0.124186  |            0.706743 | 0.00517651 |          198.038 | ITERATE           | True                 | True           | False                    | True                     |
| core_champion_replay    | core_champion_replay    | compact          |     0.312599 |    0.509456 |                  0.267388 |         0.636364 |                0.195076 |         0.484022 |          0.095405  |            0.21708  |         0.0965454 |            0.709141 | 0.00490624 |          330.873 | ITERATE           | False                | False          | False                    | True                     |
| core_conservative       | core                    | conservative     |     0.30831  |    0.51158  |                  0.263236 |         0.636364 |                0.190787 |         0.482646 |          0.0854227 |            0.199129 |         0.107019  |            0.706    | 0.00371676 |          198.846 | ITERATE           | False                | False          | False                    | False                    |
| core_balanced           | core                    | balanced         |     0.307701 |    0.517448 |                  0.258297 |         0.626263 |                0.190178 |         0.481891 |          0.088577  |            0.208421 |         0.0975669 |            0.700927 | 0.00478691 |          217.589 | ITERATE           | False                | False          | False                    | False                    |
| imdb_feature_only       | imdb_feature_only       | compact          |     0.304503 |    0.50683  |                  0.262225 |         0.636364 |                0.18698  |         0.480524 |          0.0833491 |            0.194136 |         0.0989371 |            0.704656 | 0.00446411 |          203.017 | ITERATE           | False                | True           | False                    | False                    |
| tag_genome_feature_only | tag_genome_feature_only | compact          |     0.302419 |    0.484315 |                  0.25942  |         0.626263 |                0.184897 |         0.493908 |          0.0767961 |            0.184076 |         0.0953814 |            0.701833 | 0.00385446 |          210.466 | ITERATE           | True                 | False          | False                    | False                    |

## Canonical baseline

`core_champion_replay` is the canonical promotion baseline.

## Replay interpretation

The legacy profile-selection run does not have a fully matching frozen contract and config fingerprint. Metric deltas are diagnostic only; core_champion_replay is the canonical promotion baseline.

## Champion

`combined_feature_only` with NDCG@10 `0.3220`.