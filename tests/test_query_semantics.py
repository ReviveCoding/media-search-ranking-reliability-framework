import pandas as pd

from media_search_reliability.data_ingestion.synthetic_data import generate_synthetic_movielens
from media_search_reliability.data_ingestion.build_media_catalog import build_media_catalog
from media_search_reliability.features.retrieval_features import FEATURE_COLUMNS, build_candidate_features
from media_search_reliability.query_labeling.generate_queries import generate_queries_and_labels


def _fixture(seed=17):
    movies, ratings, tags = generate_synthetic_movielens(
        n_movies=70, n_users=35, n_ratings=600, seed=seed, tag_observation_prob=0.8
    )
    catalog = build_media_catalog(movies, ratings, tags)
    queries, labels, users = generate_queries_and_labels(
        catalog, ratings, num_queries=15, candidates_per_query=30, seed=seed, label_all_catalog=True
    )
    return catalog, queries, labels, users


def test_similar_to_anchor_is_not_relevant():
    _, queries, labels, _ = _fixture()
    similar = queries[queries["query_type"] == "similar_to"]
    assert len(similar) > 0
    merged = similar[["query_id", "anchor_movie_id"]].merge(labels, on="query_id")
    self_rows = merged[merged["anchor_movie_id"] == merged["movie_id"]]
    assert len(self_rows) == len(similar)
    assert (self_rows["clean_label"] == 0).all()


def test_mood_decade_query_has_grounded_mood_target():
    catalog, queries, labels, _ = _fixture(seed=19)
    mood = queries[queries["query_type"] == "mood_decade"]
    assert len(mood) > 0
    assert mood["target_mood_tag"].astype(str).str.len().gt(0).all()
    relevant = mood[["query_id", "target_mood_tag"]].merge(labels[labels["clean_label"] >= 2], on="query_id")
    relevant = relevant.merge(catalog[["movie_id", "synthetic_tags"]], on="movie_id")
    # At least some high-relevance results should carry the requested latent mood tag.
    grounded = [tag in str(tags).split("|") for tag, tags in zip(relevant["target_mood_tag"], relevant["synthetic_tags"])]
    assert any(grounded)


def test_query_understanding_features_are_present():
    catalog, queries, labels, users = _fixture(seed=23)
    q = queries.iloc[[0]].copy()
    candidate_ids = labels[labels["query_id"] == int(q.iloc[0].query_id)]["movie_id"].head(5)
    candidates = pd.DataFrame({
        "query_id": int(q.iloc[0].query_id),
        "movie_id": candidate_ids,
        "bm25_score": 0.1,
        "dense_score": 0.2,
        "hybrid_score": 0.15,
    })
    features = build_candidate_features(candidates, q, catalog, labels, users)
    expected = {"mood_overlap", "anchor_match", "query_is_similar_to", "query_is_visual", "query_is_mood_decade"}
    assert expected.issubset(features.columns)
    assert expected.issubset(FEATURE_COLUMNS)
