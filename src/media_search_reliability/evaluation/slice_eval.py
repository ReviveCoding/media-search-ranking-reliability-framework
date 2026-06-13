from __future__ import annotations

import numpy as np
import pandas as pd
from media_search_reliability.evaluation.ranking_metrics import metrics_at_k


def _item_slice_recall_at_k(
    ranked_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    flag_col: str,
    score_col: str,
    k: int,
    positive_label_min: int,
) -> tuple[float, float, float, int]:
    """Return raw recall, capacity-adjusted efficiency, and query hit rate."""
    recalls, efficiencies, hit_rates = [], [], []
    ranked_groups = {int(q): g.sort_values([score_col, "movie_id"], ascending=[False, True]) for q, g in ranked_df.groupby("query_id")}
    for query_id, truth in ground_truth.groupby("query_id"):
        relevant_slice = truth[(truth["label"] >= positive_label_min) & (truth[flag_col].astype(int) == 1)]
        if relevant_slice.empty:
            continue
        top_ids = set(ranked_groups.get(int(query_id), pd.DataFrame()).head(k).get("movie_id", pd.Series(dtype=int)).astype(int).tolist())
        hits = int(relevant_slice["movie_id"].astype(int).isin(top_ids).sum())
        total_relevant = int(len(relevant_slice))
        recalls.append(float(hits / total_relevant))
        efficiencies.append(float(hits / min(k, total_relevant)))
        hit_rates.append(float(hits > 0))
    return (
        float(np.mean(recalls)) if recalls else 0.0,
        float(np.mean(efficiencies)) if efficiencies else 0.0,
        float(np.mean(hit_rates)) if hit_rates else 0.0,
        len(recalls),
    )


def evaluate_slices(
    df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    score_col: str = "ranker_score",
    k: int = 10,
    positive_label_min: int = 2,
    min_queries_for_claims: int = 5,
) -> pd.DataFrame:
    """Evaluate query slices and full-ranking item slices.

    Query-type slices use the full ranking for selected queries. Long-tail and
    cold-start metrics measure relevant sliced-item hits inside the full top-k.
    Raw recall is retained for transparency, while recall efficiency adjusts the
    denominator to the maximum number of relevant items that can fit in top-k.
    """
    def support_status(n_queries: int) -> str:
        return "SUFFICIENT" if int(n_queries) >= int(min_queries_for_claims) else "LOW_SUPPORT"

    rows = []
    all_metrics = metrics_at_k(df, score_col=score_col, k=k, positive_label_min=positive_label_min, ground_truth=ground_truth)
    all_query_count = int(df["query_id"].nunique())
    rows.append({**all_metrics, "slice": "all", "metric_type": "query_ranking", "num_rows": len(df), "num_queries": all_query_count, "claim_support": support_status(all_query_count)})

    for query_type in sorted(df.get("query_type", pd.Series(dtype=str)).dropna().astype(str).unique()):
        qids = set(df.loc[df["query_type"].astype(str) == query_type, "query_id"].unique())
        sub = df[df["query_id"].isin(qids)]
        truth = ground_truth[ground_truth["query_id"].isin(qids)]
        metrics = metrics_at_k(sub, score_col=score_col, k=k, positive_label_min=positive_label_min, ground_truth=truth)
        rows.append({**metrics, "slice": f"query_type:{query_type}", "metric_type": "query_ranking", "num_rows": len(sub), "num_queries": len(qids), "claim_support": support_status(len(qids))})

    for slice_name, flag_col in [("long_tail", "slice_long_tail"), ("cold_start", "slice_cold_start")]:
        if flag_col not in ground_truth:
            continue
        recall, efficiency, hit_rate, n_queries = _item_slice_recall_at_k(df, ground_truth, flag_col, score_col, k, positive_label_min)
        rows.append({
            f"ndcg_at_{k}": np.nan,
            f"mrr_at_{k}": np.nan,
            f"recall_at_{k}": recall,
            f"recall_efficiency_at_{k}": efficiency,
            f"precision_at_{k}": np.nan,
            f"hit_rate_at_{k}": hit_rate,
            "slice": slice_name,
            "metric_type": "relevant_item_recall",
            "num_rows": int((ground_truth[flag_col].astype(int) == 1).sum()),
            "num_queries": n_queries,
            "claim_support": support_status(n_queries),
        })
    return pd.DataFrame(rows)
