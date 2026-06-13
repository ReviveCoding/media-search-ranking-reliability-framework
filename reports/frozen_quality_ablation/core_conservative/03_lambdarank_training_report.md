# LambdaRank Training Report

## Ranker backend

lightgbm-lambdarank-gpu

## Feature columns

bm25_score, dense_score, hybrid_score, specialized_score, anchor_dense_score, anchor_metadata_score, personalized_dense_score, genre_overlap, tag_overlap, mood_overlap, tag_genome_tag_overlap, tag_genome_mood_overlap, imdb_genre_overlap, user_genre_affinity, user_tag_affinity, tag_genome_user_affinity, user_history_confidence, anchor_genre_jaccard, anchor_tag_jaccard, anchor_tag_genome_jaccard, anchor_year_similarity, anchor_rating_similarity, movie_popularity_log, movie_recency, rating_mean, tag_genome_coverage, imdb_rating, imdb_rating_centered, imdb_votes_log, imdb_vote_confidence, imdb_runtime_scaled, imdb_coverage, long_tail_flag, cold_start_flag, decade_match, anchor_match, query_is_similar_to, query_is_personalized, query_is_visual, query_is_mood_decade

## Train/validation/test queries

324 / 77 / 99

## Training-only injected rows

3671

## Calibration method

isotonic

## Split strategy

group-aware-best-of-random-candidates

## Anchor leakage free

True

## Personalized-user leakage free

True
