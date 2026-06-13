# Slice Reliability Report

## Methodology

Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.

## Claim-support rule

Slices with fewer than 5 held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.

## Held-out slices

|   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 | slice                   | metric_type          |   num_rows |   num_queries | claim_support   |
|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|:------------------------|:---------------------|-----------:|--------------:|:----------------|
|    0.304503  |    0.50683  |     0.193115   |                0.262225   |         0.182828  |        0.636364  | all                     | query_ranking        |      13740 |            99 | SUFFICIENT      |
|    0.480524  |    0.842105 |     0.209296   |                0.28797    |         0.231579  |        0.842105  | query_type:genre_tag    | query_ranking        |       2145 |            19 | SUFFICIENT      |
|    0.0833491 |    0.166667 |     0.0213223  |                0.075      |         0.075     |        0.45      | query_type:mood_decade  | query_ranking        |       2395 |            20 | SUFFICIENT      |
|    0.194136  |    0.3975   |     0.0346892  |                0.18       |         0.18      |        0.6       | query_type:personalized | query_ranking        |       2808 |            20 | SUFFICIENT      |
|    0.0989371 |    0.185374 |     0.026231   |                0.0761905  |         0.0761905 |        0.333333  | query_type:similar_to   | query_ranking        |       4387 |            21 | SUFFICIENT      |
|    0.704656  |    1        |     0.708985   |                0.725731   |         0.368421  |        1         | query_type:visual_query | query_ranking        |       2005 |            19 | SUFFICIENT      |
|  nan         |  nan        |     0.00352113 |                0.00352113 |       nan         |        0.0140845 | long_tail               | relevant_item_recall |       1742 |            71 | SUFFICIENT      |
|  nan         |  nan        |     0          |                0          |       nan         |        0         | cold_start              | relevant_item_recall |        925 |            54 | SUFFICIENT      |

## Retrieval-only scenario diagnostic

{
  "visual_query_hybrid_ndcg_all": 0.24057777758054505,
  "visual_query_count_all": 100
}
