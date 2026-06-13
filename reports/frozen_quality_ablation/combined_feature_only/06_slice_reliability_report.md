# Slice Reliability Report

## Methodology

Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.

## Claim-support rule

Slices with fewer than 5 held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.

## Held-out slices

|   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 | slice                   | metric_type          |   num_rows |   num_queries | claim_support   |
|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|:------------------------|:---------------------|-----------:|--------------:|:----------------|
|     0.322034 |    0.546196 |      0.195613  |                 0.265705  |         0.183838  |         0.666667 | all                     | query_ranking        |      13740 |            99 | SUFFICIENT      |
|     0.498418 |    0.842105 |      0.210561  |                 0.288555  |         0.231579  |         0.842105 | query_type:genre_tag    | query_ranking        |       2145 |            19 | SUFFICIENT      |
|     0.102074 |    0.25631  |      0.0194704 |                 0.07      |         0.07      |         0.45     | query_type:mood_decade  | query_ranking        |       2395 |            20 | SUFFICIENT      |
|     0.216694 |    0.474306 |      0.0349177 |                 0.175     |         0.175     |         0.65     | query_type:personalized | query_ranking        |       2808 |            20 | SUFFICIENT      |
|     0.124186 |    0.212434 |      0.0453334 |                 0.101587  |         0.0904762 |         0.428571 | query_type:similar_to   | query_ranking        |       4387 |            21 | SUFFICIENT      |
|     0.706743 |    1        |      0.701329  |                 0.725731  |         0.368421  |         1        | query_type:visual_query | query_ranking        |       2005 |            19 | SUFFICIENT      |
|   nan        |  nan        |      0.0211268 |                 0.0211268 |       nan         |         0.056338 | long_tail               | relevant_item_recall |       1742 |            71 | SUFFICIENT      |
|   nan        |  nan        |      0.0277778 |                 0.0277778 |       nan         |         0.037037 | cold_start              | relevant_item_recall |        925 |            54 | SUFFICIENT      |

## Retrieval-only scenario diagnostic

{
  "visual_query_hybrid_ndcg_all": 0.24057777758054505,
  "visual_query_count_all": 100
}
