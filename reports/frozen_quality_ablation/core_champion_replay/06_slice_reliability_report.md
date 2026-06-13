# Slice Reliability Report

## Methodology

Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.

## Claim-support rule

Slices with fewer than 5 held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.

## Held-out slices

|   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 | slice                   | metric_type          |   num_rows |   num_queries | claim_support   |
|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|:------------------------|:---------------------|-----------:|--------------:|:----------------|
|    0.312599  |    0.509456 |     0.196202   |                0.267388   |         0.187879  |        0.636364  | all                     | query_ranking        |      13740 |            99 | SUFFICIENT      |
|    0.484022  |    0.842105 |     0.217536   |                0.299081   |         0.242105  |        0.842105  | query_type:genre_tag    | query_ranking        |       2145 |            19 | SUFFICIENT      |
|    0.095405  |    0.2175   |     0.0213223  |                0.075      |         0.075     |        0.45      | query_type:mood_decade  | query_ranking        |       2395 |            20 | SUFFICIENT      |
|    0.21708   |    0.427917 |     0.0366833  |                0.185      |         0.185     |        0.6       | query_type:personalized | query_ranking        |       2808 |            20 | SUFFICIENT      |
|    0.0965454 |    0.14418  |     0.0270998  |                0.0761905  |         0.0761905 |        0.333333  | query_type:similar_to   | query_ranking        |       4387 |            21 | SUFFICIENT      |
|    0.709141  |    0.973684 |     0.713769   |                0.736257   |         0.378947  |        1         | query_type:visual_query | query_ranking        |       2005 |            19 | SUFFICIENT      |
|  nan         |  nan        |     0.0140845  |                0.0140845  |       nan         |        0.0422535 | long_tail               | relevant_item_recall |       1742 |            71 | SUFFICIENT      |
|  nan         |  nan        |     0.00925926 |                0.00925926 |       nan         |        0.0185185 | cold_start              | relevant_item_recall |        925 |            54 | SUFFICIENT      |

## Retrieval-only scenario diagnostic

{
  "visual_query_hybrid_ndcg_all": 0.24057777758054505,
  "visual_query_count_all": 100
}
