import numpy as np
import pandas as pd

from media_search_reliability.evaluation.ranking_metrics import metrics_at_k
from media_search_reliability.evaluation.slice_eval import evaluate_slices
from media_search_reliability.pipeline import _split_queries
from media_search_reliability.retrieval.faiss_index import VectorIndex


def test_vector_index_caps_top_k_to_catalog_size():
    embeddings = np.eye(4, dtype=np.float32)
    index = VectorIndex(use_gpu_if_available=False).fit(embeddings, [10, 11, 12, 13])
    result = index.search(embeddings[:1], top_k=100)[0]
    assert len(result) == 4


def test_external_ground_truth_prevents_optimistic_recall():
    ranked = pd.DataFrame({
        "query_id": [1], "movie_id": [10], "label": [3], "score": [1.0],
    })
    truth = pd.DataFrame({
        "query_id": [1, 1, 1, 1], "movie_id": [10, 11, 12, 13], "label": [3, 2, 2, 2],
    })
    metrics = metrics_at_k(ranked, "score", k=1, ground_truth=truth, positive_label_min=2)
    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_efficiency_at_1"] == 1.0


def test_long_tail_slice_uses_full_ranking_not_filtered_reranking():
    ranked = pd.DataFrame({
        "query_id": [1, 1], "movie_id": [10, 11], "label": [3, 2], "score": [1.0, 0.1], "query_type": ["genre_tag", "genre_tag"],
    })
    truth = pd.DataFrame({
        "query_id": [1, 1], "movie_id": [10, 11], "label": [3, 2],
        "slice_long_tail": [0, 1], "slice_cold_start": [0, 1],
    })
    slices = evaluate_slices(ranked, truth, score_col="score", k=1, positive_label_min=2)
    long_tail = slices.loc[slices["slice"] == "long_tail", "recall_at_1"].iloc[0]
    assert long_tail == 0.0


def test_stratified_split_covers_each_query_type():
    queries = pd.DataFrame({
        "query_id": list(range(20)),
        "query_type": [q for q in ["a", "b", "c", "d"] for _ in range(5)],
    })
    train, val, test = _split_queries(queries, test_size=0.2, val_size=0.2, seed=5)
    for split in [train, val, test]:
        covered = set(queries[queries["query_id"].isin(split)]["query_type"])
        assert covered == {"a", "b", "c", "d"}
