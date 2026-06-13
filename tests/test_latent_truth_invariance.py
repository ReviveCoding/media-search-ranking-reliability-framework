import pandas as pd

from media_search_reliability.data_ingestion.synthetic_data import generate_synthetic_movielens
from media_search_reliability.data_ingestion.build_media_catalog import build_media_catalog
from media_search_reliability.query_labeling.generate_queries import generate_queries_and_labels


def _scenario(rating_noise: float, preference_strength: float, exploration_rate: float, tag_observation_prob: float):
    movies, ratings, tags = generate_synthetic_movielens(
        n_movies=70,
        n_users=40,
        n_ratings=700,
        seed=1234,
        rating_noise=rating_noise,
        preference_strength=preference_strength,
        exploration_rate=exploration_rate,
        tag_observation_prob=tag_observation_prob,
    )
    catalog = build_media_catalog(movies, ratings, tags)
    return generate_queries_and_labels(
        catalog,
        ratings,
        num_queries=25,
        candidates_per_query=20,
        seed=5678,
        label_all_catalog=True,
        label_noise=0.0,
    )


def test_clean_queries_and_labels_are_invariant_to_observation_noise():
    nominal_queries, nominal_labels, _ = _scenario(0.6, 1.4, 0.10, 0.55)
    noisy_queries, noisy_labels, _ = _scenario(1.5, 0.7, 0.30, 0.20)

    query_cols = [
        "query_id", "query", "query_type", "anchor_movie_id", "user_id",
        "target_genre", "target_tag", "target_mood_tag", "target_decade",
    ]
    pd.testing.assert_frame_equal(
        nominal_queries[query_cols].sort_values("query_id").reset_index(drop=True),
        noisy_queries[query_cols].sort_values("query_id").reset_index(drop=True),
        check_exact=True,
    )

    key = ["query_id", "movie_id"]
    nominal_clean = nominal_labels[key + ["clean_label"]].sort_values(key).reset_index(drop=True)
    noisy_clean = noisy_labels[key + ["clean_label"]].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(nominal_clean, noisy_clean, check_exact=True)

    nominal_observed = nominal_labels[key + ["label"]].sort_values(key).reset_index(drop=True)
    noisy_observed = noisy_labels[key + ["label"]].sort_values(key).reset_index(drop=True)
    assert not nominal_observed["label"].equals(noisy_observed["label"])
