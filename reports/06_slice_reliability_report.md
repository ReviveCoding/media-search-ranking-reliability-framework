# Slice Reliability Report

## Methodology

Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.

## Claim-support rule

Slices with fewer than 10 held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.

## Held-out slices

|   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 | slice                   | metric_type          |   num_rows |   num_queries | claim_support   |
|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|:------------------------|:---------------------|-----------:|--------------:|:----------------|
|    0.165901  |    0.388003 |      0.0627936 |                 0.137381  |         0.121667  |         0.583333 | all                     | query_ranking        |      12064 |            60 | SUFFICIENT      |
|    0.237069  |    0.6      |      0.0528448 |                 0.124286  |         0.12      |         0.6      | query_type:genre_tag    | query_ranking        |       1630 |            10 | SUFFICIENT      |
|    0.0527763 |    0.107228 |      0.023911  |                 0.0642857 |         0.0642857 |         0.5      | query_type:mood_decade  | query_ranking        |       2301 |            14 | SUFFICIENT      |
|    0.0985041 |    0.234259 |      0.0314674 |                 0.108333  |         0.108333  |         0.416667 | query_type:personalized | query_ranking        |       2756 |            12 | SUFFICIENT      |
|    0.109809  |    0.288988 |      0.0265976 |                 0.0833333 |         0.0833333 |         0.583333 | query_type:similar_to   | query_ranking        |       3387 |            12 | SUFFICIENT      |
|    0.36206   |    0.791667 |      0.18397   |                 0.316667  |         0.241667  |         0.833333 | query_type:visual_query | query_ranking        |       1990 |            12 | SUFFICIENT      |
|  nan         |  nan        |      0.0605413 |                 0.0605413 |       nan         |         0.128205 | long_tail               | relevant_item_recall |        621 |            39 | SUFFICIENT      |
|  nan         |  nan        |      0         |                 0         |       nan         |         0        | cold_start              | relevant_item_recall |        117 |            17 | SUFFICIENT      |

## Retrieval-only scenario diagnostic

{
  "visual_query_hybrid_ndcg_all": 0.12181785469126974,
  "visual_query_count_all": 60
}
