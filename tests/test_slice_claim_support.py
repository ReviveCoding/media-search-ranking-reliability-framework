import pandas as pd

from media_search_reliability.evaluation.slice_eval import evaluate_slices


def test_small_query_slice_is_marked_low_support():
    ranked = pd.DataFrame([
        {"query_id": 1, "movie_id": 1, "score": 1.0, "label": 3, "query_type": "visual_query"},
        {"query_id": 1, "movie_id": 2, "score": 0.0, "label": 0, "query_type": "visual_query"},
    ])
    truth = ranked[["query_id", "movie_id", "label"]].copy()
    result = evaluate_slices(ranked, truth, score_col="score", k=1, min_queries_for_claims=5)
    row = result[result["slice"] == "query_type:visual_query"].iloc[0]
    assert row["claim_support"] == "LOW_SUPPORT"
    assert int(row["num_queries"]) == 1
