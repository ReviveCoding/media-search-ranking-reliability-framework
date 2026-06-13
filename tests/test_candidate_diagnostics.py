from __future__ import annotations

import pandas as pd

from media_search_reliability.evaluation.candidate_diagnostics import candidate_recall_diagnostics


def test_candidate_recall_diagnostics_reports_source_and_union() -> None:
    candidates = pd.DataFrame({
        "query_id": [1, 1, 1], "movie_id": [10, 11, 12],
        "bm25_score": [1.0, 0.0, 0.5], "dense_score": [0.0, 1.0, 0.2],
        "hybrid_score": [1.0, 0.9, 0.5], "specialized_score": [0.0, 0.0, 0.0],
    })
    labels = pd.DataFrame({
        "query_id": [1, 1, 1], "movie_id": [10, 11, 12], "label": [3, 2, 0],
    })
    queries = pd.DataFrame({"query_id": [1], "query_type": ["genre_tag"]})
    result = candidate_recall_diagnostics(candidates, labels, queries, k_values=(2,))
    union = result[(result["source"] == "candidate_union") & (result["k"] == 2)].iloc[0]
    assert union["mean_recall_efficiency"] == 1.0
    assert set(result["source"]) >= {"candidate_union", "bm25_score", "dense_score", "hybrid_score"}
