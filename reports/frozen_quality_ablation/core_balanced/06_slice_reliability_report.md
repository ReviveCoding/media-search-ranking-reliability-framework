# Slice Reliability Report

## Methodology

Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.

## Claim-support rule

Slices with fewer than 5 held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.

## Held-out slices

|   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 | slice                   | metric_type          |   num_rows |   num_queries | claim_support   |
|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|:------------------------|:---------------------|-----------:|--------------:|:----------------|
|    0.307701  |    0.517448 |     0.191191   |                0.258297   |         0.178788  |        0.626263  | all                     | query_ranking        |      13740 |            99 | SUFFICIENT      |
|    0.481891  |    0.815789 |     0.214748   |                0.293818   |         0.236842  |        0.842105  | query_type:genre_tag    | query_ranking        |       2145 |            19 | SUFFICIENT      |
|    0.088577  |    0.214167 |     0.0194704  |                0.07       |         0.07      |        0.45      | query_type:mood_decade  | query_ranking        |       2395 |            20 | SUFFICIENT      |
|    0.208421  |    0.463393 |     0.0328505  |                0.17       |         0.17      |        0.6       | query_type:personalized | query_ranking        |       2808 |            20 | SUFFICIENT      |
|    0.0975669 |    0.151247 |     0.0265652  |                0.0761905  |         0.0761905 |        0.285714  | query_type:similar_to   | query_ranking        |       4387 |            21 | SUFFICIENT      |
|    0.700927  |    1        |     0.697023   |                0.715205   |         0.357895  |        1         | query_type:visual_query | query_ranking        |       2005 |            19 | SUFFICIENT      |
|  nan         |  nan        |     0.00704225 |                0.00704225 |       nan         |        0.028169  | long_tail               | relevant_item_recall |       1742 |            71 | SUFFICIENT      |
|  nan         |  nan        |     0.00925926 |                0.00925926 |       nan         |        0.0185185 | cold_start              | relevant_item_recall |        925 |            54 | SUFFICIENT      |

## Retrieval-only scenario diagnostic

{
  "visual_query_hybrid_ndcg_all": 0.24057777758054505,
  "visual_query_count_all": 100
}
