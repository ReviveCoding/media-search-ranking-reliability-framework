from __future__ import annotations

import numpy as np
import pandas as pd


def _recall_efficiency(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return float(hits / max(1, min(len(relevant_ids), k)))


def candidate_recall_diagnostics(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    positive_label_min: int = 2,
    k_values: tuple[int, ...] = (50, 100),
) -> pd.DataFrame:
    truth = labels[pd.to_numeric(labels["label"], errors="coerce").fillna(0) >= positive_label_min]
    relevant_by_query = truth.groupby("query_id")["movie_id"].apply(lambda s: set(s.astype(int))).to_dict()
    query_type = queries.set_index("query_id")["query_type"].astype(str).to_dict()
    source_columns = [
        c for c in (
            "bm25_score", "dense_score", "hybrid_score", "specialized_score",
            "anchor_dense_score", "anchor_metadata_score", "personalized_dense_score",
        ) if c in candidates.columns
    ]
    rows: list[dict] = []
    for qid, group in candidates.groupby("query_id", sort=True):
        relevant = relevant_by_query.get(qid, set())
        if not relevant:
            continue
        qtype = query_type.get(qid, "unknown")
        union_ids = group["movie_id"].astype(int).tolist()
        for k in k_values:
            rows.append({
                "query_id": int(qid),
                "query_type": qtype,
                "source": "candidate_union",
                "k": int(k),
                "recall_efficiency": _recall_efficiency(union_ids, relevant, k),
                "candidate_count": int(len(group)),
                "positive_count": int(len(relevant)),
            })
        for source in source_columns:
            ranked = group.sort_values([source, "movie_id"], ascending=[False, True], kind="mergesort")
            ids = ranked.loc[pd.to_numeric(ranked[source], errors="coerce").fillna(0) > 0, "movie_id"].astype(int).tolist()
            for k in k_values:
                rows.append({
                    "query_id": int(qid),
                    "query_type": qtype,
                    "source": source,
                    "k": int(k),
                    "recall_efficiency": _recall_efficiency(ids, relevant, k),
                    "candidate_count": int(len(ids)),
                    "positive_count": int(len(relevant)),
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["query_type", "source", "k", "num_queries", "mean_recall_efficiency", "p10_recall_efficiency", "mean_candidate_count"])
    return (
        detail.groupby(["query_type", "source", "k"], as_index=False)
        .agg(
            num_queries=("query_id", "nunique"),
            mean_recall_efficiency=("recall_efficiency", "mean"),
            p10_recall_efficiency=("recall_efficiency", lambda s: float(np.nanquantile(s, 0.10))),
            mean_candidate_count=("candidate_count", "mean"),
        )
        .sort_values(["query_type", "source", "k"], kind="mergesort")
        .reset_index(drop=True)
    )
