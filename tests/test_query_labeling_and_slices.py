from media_search_reliability.data_ingestion.synthetic_data import generate_synthetic_movielens
from media_search_reliability.data_ingestion.build_media_catalog import build_media_catalog
from media_search_reliability.query_labeling.generate_queries import generate_queries_and_labels


def test_query_generation_has_stable_slices_and_graded_labels():
    movies, ratings, tags = generate_synthetic_movielens(n_movies=120, n_users=60, n_ratings=900, seed=7)
    catalog = build_media_catalog(movies, ratings, tags)
    queries, labels, _ = generate_queries_and_labels(catalog, ratings, num_queries=25, candidates_per_query=25, seed=7)

    assert set(["genre_tag", "similar_to", "personalized", "mood_decade", "visual_query"]).issubset(set(queries["query_type"]))
    assert labels["label"].between(0, 3).all()
    assert {0, 1, 2, 3}.issubset(set(labels["label"].unique()))
    assert labels.groupby("query_id")["label"].max().min() >= 2
