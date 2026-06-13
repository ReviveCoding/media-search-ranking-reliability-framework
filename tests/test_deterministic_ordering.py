import pandas as pd

from media_search_reliability.data_ingestion.build_media_catalog import build_media_catalog
from media_search_reliability.evaluation.ranking_metrics import metrics_at_k
from media_search_reliability.retrieval.bm25_retriever import BM25Retriever


def test_catalog_tag_aggregation_uses_deterministic_alphabetical_tiebreak():
    movies = pd.DataFrame({
        "movie_id": [1],
        "title": ["Example (2000)"],
        "genres": ["Drama"],
    })
    ratings = pd.DataFrame({
        "user_id": [1, 2],
        "movie_id": [1, 1],
        "rating": [4.0, 3.5],
    })
    tags = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "movie_id": [1, 1, 1, 1],
        "tag": ["zeta", "alpha", "zeta", "alpha"],
    })

    catalog = build_media_catalog(movies, ratings, tags)
    assert catalog.loc[0, "tag_text"] == "alpha|zeta"


def test_bm25_ties_are_broken_by_movie_id():
    retriever = BM25Retriever().fit(["same", "same", "same"], doc_ids=[30, 10, 20])
    assert [movie_id for movie_id, _ in retriever.search("missing", top_k=3)] == [10, 20, 30]


def test_metric_ties_are_broken_by_movie_id():
    ranked = pd.DataFrame({
        "query_id": [1, 1],
        "movie_id": [2, 1],
        "label": [0, 3],
        "score": [0.5, 0.5],
    })
    metrics = metrics_at_k(ranked, "score", k=1)
    assert metrics["mrr_at_1"] == 1.0
