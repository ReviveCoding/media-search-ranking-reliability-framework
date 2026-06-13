from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SplitResult:
    train_query_ids: set[int]
    val_query_ids: set[int]
    test_query_ids: set[int]
    diagnostics: dict


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}
        self.rank = {int(v): 0 for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_leakage_groups(queries: pd.DataFrame) -> pd.Series:
    """Build connected components that prevent anchor and personalized-user leakage.

    Queries sharing an anchor movie are kept in one split. Personalized queries
    sharing a user are also kept in one split. Union-find handles transitive links.
    """
    qids = queries["query_id"].astype(int).tolist()
    uf = _UnionFind(qids)

    if "anchor_movie_id" in queries:
        for _, group in queries[pd.to_numeric(queries["anchor_movie_id"], errors="coerce").fillna(-1).astype(int) >= 0].groupby("anchor_movie_id"):
            values = group["query_id"].astype(int).tolist()
            for value in values[1:]:
                uf.union(values[0], value)

    if {"query_type", "user_id"}.issubset(queries.columns):
        personalized = queries[queries["query_type"].astype(str) == "personalized"]
        for _, group in personalized.groupby("user_id"):
            values = group["query_id"].astype(int).tolist()
            for value in values[1:]:
                uf.union(values[0], value)

    roots = {qid: uf.find(qid) for qid in qids}
    compact = {root: idx for idx, root in enumerate(sorted(set(roots.values())))}
    return queries["query_id"].astype(int).map(lambda qid: compact[roots[int(qid)]])


def _split_score(frame: pd.DataFrame, holdout_index: np.ndarray, holdout_fraction: float) -> float:
    holdout = frame.iloc[holdout_index]
    size_error = abs(len(holdout) / max(1, len(frame)) - holdout_fraction)
    overall = frame["query_type"].astype(str).value_counts(normalize=True)
    observed = holdout["query_type"].astype(str).value_counts(normalize=True)
    distribution_error = float(sum(abs(observed.get(label, 0.0) - overall.get(label, 0.0)) for label in overall.index))
    missing_types = int(sum(label not in observed.index for label in overall.index))
    return 2.0 * size_error + distribution_error + 5.0 * missing_types


def _best_group_holdout(frame: pd.DataFrame, holdout_fraction: float, seed: int, n_candidates: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) < 3 or frame["split_group"].nunique() < 2:
        raise ValueError("Not enough independent groups for a group-aware split.")
    splitter = GroupShuffleSplit(n_splits=n_candidates, test_size=holdout_fraction, random_state=seed)
    best = None
    best_score = float("inf")
    dummy = np.zeros(len(frame))
    for train_index, holdout_index in splitter.split(dummy, frame["query_type"], groups=frame["split_group"]):
        score = _split_score(frame, holdout_index, holdout_fraction)
        if score < best_score:
            best_score = score
            best = (train_index, holdout_index)
    if best is None:
        raise RuntimeError("Unable to produce a group-aware split.")
    return best


def split_queries_group_aware(
    queries: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int = 42,
) -> SplitResult:
    if not 0 < test_size < 1 or not 0 <= val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be valid fractions with sum < 1.")

    frame = queries.copy().reset_index(drop=True)
    if "query_type" not in frame:
        frame["query_type"] = "all"
    frame["split_group"] = build_leakage_groups(frame).values

    try:
        remaining_index, test_index = _best_group_holdout(frame, test_size, seed)
        remaining = frame.iloc[remaining_index].reset_index(drop=True)
        relative_val = val_size / (1.0 - test_size)
        train_rel, val_rel = _best_group_holdout(remaining, relative_val, seed + 1)
        train_ids = set(remaining.iloc[train_rel]["query_id"].astype(int))
        val_ids = set(remaining.iloc[val_rel]["query_id"].astype(int))
        test_ids = set(frame.iloc[test_index]["query_id"].astype(int))
        strategy = "group-aware-best-of-random-candidates"
    except Exception:
        # Deterministic query-type-stratified fallback for tiny edge cases.
        rng = np.random.default_rng(seed)
        train_ids, val_ids, test_ids = set(), set(), set()
        for _, group in frame.groupby("query_type"):
            qids = group["query_id"].astype(int).unique().copy()
            rng.shuffle(qids)
            n = len(qids)
            n_test = max(1, int(round(n * test_size))) if n >= 3 else int(n == 2)
            n_val = max(1, int(round(n * val_size))) if n >= 4 else 0
            while n_test + n_val >= n and n_val > 0:
                n_val -= 1
            test_ids.update(map(int, qids[:n_test]))
            val_ids.update(map(int, qids[n_test:n_test + n_val]))
            train_ids.update(map(int, qids[n_test + n_val:]))
        strategy = "query-type-stratified-fallback"

    diagnostics = split_leakage_diagnostics(frame, train_ids, val_ids, test_ids)
    diagnostics["strategy"] = strategy
    diagnostics["split_sizes"] = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    return SplitResult(train_ids, val_ids, test_ids, diagnostics)


def _pairwise_overlap(values_by_split: dict[str, set[int]]) -> dict[str, int]:
    names = list(values_by_split)
    return {
        f"{left}_{right}": len(values_by_split[left].intersection(values_by_split[right]))
        for idx, left in enumerate(names)
        for right in names[idx + 1:]
    }


def split_leakage_diagnostics(
    queries: pd.DataFrame,
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
) -> dict:
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    anchors: dict[str, set[int]] = {}
    users: dict[str, set[int]] = {}
    query_type_counts: dict[str, dict[str, int]] = {}
    for name, ids in split_ids.items():
        subset = queries[queries["query_id"].astype(int).isin(ids)]
        if "anchor_movie_id" in subset:
            anchors[name] = set(pd.to_numeric(subset["anchor_movie_id"], errors="coerce").dropna().astype(int)) - {-1}
        else:
            anchors[name] = set()
        if {"query_type", "user_id"}.issubset(subset.columns):
            personalized = subset[subset["query_type"].astype(str) == "personalized"]
            users[name] = set(pd.to_numeric(personalized["user_id"], errors="coerce").dropna().astype(int))
        else:
            users[name] = set()
        query_type_counts[name] = subset["query_type"].astype(str).value_counts().sort_index().astype(int).to_dict()

    anchor_overlap = _pairwise_overlap(anchors)
    personalized_user_overlap = _pairwise_overlap(users)
    return {
        "anchor_overlap": anchor_overlap,
        "personalized_user_overlap": personalized_user_overlap,
        "query_type_counts": query_type_counts,
        "anchor_leakage_free": all(value == 0 for value in anchor_overlap.values()),
        "personalized_user_leakage_free": all(value == 0 for value in personalized_user_overlap.values()),
    }
