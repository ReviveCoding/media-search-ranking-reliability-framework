# Slice Reliability Report

## Methodology

Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.

## Claim-support rule

Slices with fewer than 5 held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.

## Held-out slices

|   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 | slice                   | metric_type          |   num_rows |   num_queries | claim_support   |
|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|:------------------------|:---------------------|-----------:|--------------:|:----------------|
|     0.325471 |    0.552902 |      0.195935  |                 0.268735  |         0.186869  |        0.656566  | all                     | query_ranking        |      13740 |            99 | SUFFICIENT      |
|     0.492995 |    0.842105 |      0.210561  |                 0.288555  |         0.231579  |        0.842105  | query_type:genre_tag    | query_ranking        |       2145 |            19 | SUFFICIENT      |
|     0.109212 |    0.264167 |      0.0213223 |                 0.075     |         0.075     |        0.45      | query_type:mood_decade  | query_ranking        |       2395 |            20 | SUFFICIENT      |
|     0.23541  |    0.495833 |      0.0369304 |                 0.19      |         0.19      |        0.6       | query_type:personalized | query_ranking        |       2808 |            20 | SUFFICIENT      |
|     0.123792 |    0.216062 |      0.0453334 |                 0.101587  |         0.0904762 |        0.428571  | query_type:similar_to   | query_ranking        |       4387 |            21 | SUFFICIENT      |
|     0.703298 |    1        |      0.698937  |                 0.720468  |         0.363158  |        1         | query_type:visual_query | query_ranking        |       2005 |            19 | SUFFICIENT      |
|   nan        |  nan        |      0.0140845 |                 0.0140845 |       nan         |        0.0422535 | long_tail               | relevant_item_recall |       1742 |            71 | SUFFICIENT      |
|   nan        |  nan        |      0.0277778 |                 0.0277778 |       nan         |        0.037037  | cold_start              | relevant_item_recall |        925 |            54 | SUFFICIENT      |

## Retrieval-only scenario diagnostic

{
  "visual_query_hybrid_ndcg_all": 0.24057777758054505,
  "visual_query_count_all": 100
}
