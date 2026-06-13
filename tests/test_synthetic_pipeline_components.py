from media_search_reliability.data_ingestion.synthetic_data import generate_synthetic_movielens
from media_search_reliability.data_ingestion.build_media_catalog import build_media_catalog
from media_search_reliability.query_labeling.generate_queries import generate_queries_and_labels


def test_synthetic_query_labels():
    movies, ratings, tags = generate_synthetic_movielens(n_movies=80, n_users=40, n_ratings=500, seed=1)
    catalog = build_media_catalog(movies, ratings, tags)
    queries, labels, users = generate_queries_and_labels(catalog, ratings, num_queries=12, candidates_per_query=20, seed=1)
    assert len(queries) == 12
    assert labels["label"].between(0, 3).all()
    assert labels["query_id"].nunique() == 12
