import pandas as pd
from media_search_reliability.evaluation.ranking_metrics import metrics_at_k


def test_metrics_at_k_basic():
    df = pd.DataFrame({
        "query_id": [1,1,1,2,2,2],
        "label": [3,0,1,0,2,0],
        "score": [0.9,0.2,0.1,0.1,0.8,0.2],
    })
    m = metrics_at_k(df, "score", k=2)
    assert 0 <= m["ndcg_at_2"] <= 1
    assert m["recall_at_2"] > 0
