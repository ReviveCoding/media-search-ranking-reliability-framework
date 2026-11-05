import numpy as np
import pandas as pd

from media_search_reliability.evaluation.ranking_metrics import (
    evaluate_variants,
    metrics_at_k,
)


def test_external_judgments_keep_missing_queries_and_tie_breaks():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "movie_id": [20, 10, 30, 40],
            "label": [0, 3, 2, 0],
            "score": [0.9, 0.9, 0.2, 0.1],
        }
    )
    truth = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 3],
            "movie_id": [10, 20, 99, 30, 40, 50],
            "label": [3, 0, 2, 2, 0, 3],
        }
    )
    result = evaluate_variants(frame, ["score"], [1, 2, 7], ground_truth=truth).iloc[0]
    assert result["recall_at_1"] == 0.5
    assert result["ndcg_at_1"] == 2 / 3
    expected = ((7 / (7 + 3 / np.log2(3))) + 1) / 3
    assert result["ndcg_at_2"] == expected
    for k in [1, 2, 7]:
        for metric, value in metrics_at_k(
            frame.iloc[::-1], "score", k=k, ground_truth=truth
        ).items():
            assert result[metric] == value


def test_empty_k_values_do_not_access_score_columns():
    result = evaluate_variants(pd.DataFrame(), ["missing"], k_values=[])
    assert result.to_dict("records") == [{"variant": "missing"}]
