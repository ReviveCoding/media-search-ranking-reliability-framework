import pandas as pd

from media_search_reliability.evaluation.splitting import split_queries_group_aware


def test_group_aware_split_prevents_anchor_and_personalized_user_leakage():
    rows = []
    query_types = ["genre_tag", "similar_to", "personalized", "mood_decade", "visual_query"]
    for query_id in range(75):
        query_type = query_types[query_id % len(query_types)]
        rows.append({
            "query_id": query_id,
            "query_type": query_type,
            "anchor_movie_id": query_id // 2,
            "user_id": query_id // 3 if query_type == "personalized" else query_id,
        })
    queries = pd.DataFrame(rows)
    result = split_queries_group_aware(queries, test_size=0.2, val_size=0.15, seed=11)
    assert result.diagnostics["anchor_leakage_free"] is True
    assert result.diagnostics["personalized_user_leakage_free"] is True
    assert result.train_query_ids
    assert result.val_query_ids
    assert result.test_query_ids
