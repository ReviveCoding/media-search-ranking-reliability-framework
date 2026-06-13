# Candidate Recall Diagnostics

## Purpose

Separates candidate-generation misses from reranker ordering errors on the frozen held-out benchmark.

## Metrics

| query_type   | source                   |   k |   num_queries |   mean_recall_efficiency |   p10_recall_efficiency |   mean_candidate_count |
|:-------------|:-------------------------|----:|--------------:|-------------------------:|------------------------:|-----------------------:|
| genre_tag    | anchor_dense_score       |  50 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | anchor_dense_score       | 100 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | anchor_metadata_score    |  50 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | anchor_metadata_score    | 100 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | bm25_score               |  50 |            19 |                 0.198492 |                       0 |                 60     |
| genre_tag    | bm25_score               | 100 |            19 |                 0.211282 |                       0 |                 60     |
| genre_tag    | candidate_union          |  50 |            19 |                 0.198492 |                       0 |                112.895 |
| genre_tag    | candidate_union          | 100 |            19 |                 0.229082 |                       0 |                112.895 |
| genre_tag    | dense_score              |  50 |            19 |                 0.142825 |                       0 |                 50     |
| genre_tag    | dense_score              | 100 |            19 |                 0.142825 |                       0 |                 50     |
| genre_tag    | hybrid_score             |  50 |            19 |                 0.183459 |                       0 |                 70     |
| genre_tag    | hybrid_score             | 100 |            19 |                 0.1976   |                       0 |                 70     |
| genre_tag    | personalized_dense_score |  50 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | personalized_dense_score | 100 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | specialized_score        |  50 |            19 |                 0        |                       0 |                  0     |
| genre_tag    | specialized_score        | 100 |            19 |                 0        |                       0 |                  0     |
| mood_decade  | anchor_dense_score       |  50 |            20 |                 0        |                       0 |                  0     |
| mood_decade  | anchor_dense_score       | 100 |            20 |                 0        |                       0 |                  0     |
| mood_decade  | anchor_metadata_score    |  50 |            20 |                 0        |                       0 |                  0     |
| mood_decade  | anchor_metadata_score    | 100 |            20 |                 0        |                       0 |                  0     |

## Interpretation

Improve candidate sources when recall efficiency is low; tune LambdaRank only when relevant items are already present in the pool.
