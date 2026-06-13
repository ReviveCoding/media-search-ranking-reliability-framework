from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(labels):
    labels = np.asarray(labels, dtype=float)
    gains = (2 ** labels - 1)
    discounts = np.log2(np.arange(len(labels)) + 2)
    return float(np.sum(gains / discounts))


def metrics_at_k(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    k: int = 10,
    positive_label_min: int = 2,
    ground_truth: pd.DataFrame | None = None,
) -> dict:
    """Compute ranking metrics using an optional external judgment set.

    When ``ground_truth`` is supplied, recall denominators and ideal DCG are
    computed from all judged candidates, not just the retrieved subset. This
    avoids the optimistic candidate-injection bias common in small demos.
    """
    if ground_truth is not None and len(ground_truth):
        query_ids = sorted(set(ground_truth["query_id"].unique()).union(df["query_id"].unique()))
        gt_groups = {int(q): g for q, g in ground_truth.groupby("query_id")}
    else:
        query_ids = sorted(df["query_id"].unique())
        gt_groups = {}

    ranked_groups = {int(q): g for q, g in df.groupby("query_id")}
    ndcgs, mrrs, recalls, recall_efficiencies, precisions, hit_rates = [], [], [], [], [], []
    for query_id in query_ids:
        ranked = ranked_groups.get(int(query_id), pd.DataFrame(columns=df.columns))
        if len(ranked):
            sort_cols = [score_col] + (["movie_id"] if "movie_id" in ranked.columns else [])
            ascending = [False] + ([True] if "movie_id" in ranked.columns else [])
            ranked = ranked.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        top_labels = ranked[label_col].astype(int).values[:k] if len(ranked) else np.array([], dtype=int)

        if int(query_id) in gt_groups:
            truth_labels = gt_groups[int(query_id)][label_col].astype(int).values
        else:
            truth_labels = ranked[label_col].astype(int).values if len(ranked) else np.array([], dtype=int)

        ideal = np.sort(truth_labels)[::-1][:k]
        ideal_dcg = _dcg(ideal)
        ndcgs.append(_dcg(top_labels) / ideal_dcg if ideal_dcg > 0 else 0.0)

        positives = top_labels >= positive_label_min
        mrrs.append(1.0 / (int(np.argmax(positives)) + 1) if positives.any() else 0.0)
        total_positive = int((truth_labels >= positive_label_min).sum())
        retrieved_positive = int((top_labels >= positive_label_min).sum())
        recalls.append(retrieved_positive / total_positive if total_positive > 0 else 0.0)
        max_retrievable = min(k, total_positive)
        recall_efficiencies.append(retrieved_positive / max_retrievable if max_retrievable > 0 else 0.0)
        precisions.append(retrieved_positive / k)
        hit_rates.append(float(retrieved_positive > 0))

    return {
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr_at_{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"recall_efficiency_at_{k}": float(np.mean(recall_efficiencies)) if recall_efficiencies else 0.0,
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"hit_rate_at_{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
    }


def evaluate_variants(
    df: pd.DataFrame,
    score_cols: list[str],
    k_values=(5, 10),
    positive_label_min: int = 2,
    ground_truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for score_col in score_cols:
        row = {"variant": score_col}
        for k in k_values:
            row.update(metrics_at_k(
                df,
                score_col=score_col,
                k=k,
                positive_label_min=positive_label_min,
                ground_truth=ground_truth,
            ))
        rows.append(row)
    return pd.DataFrame(rows)
