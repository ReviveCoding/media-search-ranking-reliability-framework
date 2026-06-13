from __future__ import annotations

import pandas as pd

from media_search_reliability.evaluation.frozen_manifest import (
    catalog_fingerprint,
    load_frozen_manifest,
    save_frozen_manifest,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame({
        "movie_id": [1, 2],
        "clean_title": ["A", "B"],
        "year": [2000, 2001],
    })


def test_frozen_manifest_roundtrip(tmp_path) -> None:
    catalog = _catalog()
    queries = pd.DataFrame({"query_id": [1], "query": ["movie"], "query_type": ["genre_tag"]})
    labels = pd.DataFrame({"query_id": [1, 1], "movie_id": [1, 2], "label": [3, 0]})
    users = pd.DataFrame({"user_id": [0], "preferred_genres": ["Drama"], "preferred_tags": [""]})
    root = save_frozen_manifest(
        tmp_path,
        queries=queries,
        labels=labels,
        user_context=users,
        train_query_ids=[1],
        val_query_ids=[],
        test_query_ids=[],
        catalog=catalog,
        seed=42,
        data_source="fixture",
    )
    loaded = load_frozen_manifest(root, catalog=catalog)
    assert loaded.train_query_ids == [1]
    assert loaded.metadata["catalog_fingerprint"] == catalog_fingerprint(catalog)
    assert loaded.labels["label"].tolist() == [3, 0]


def test_frozen_manifest_rejects_catalog_drift(tmp_path) -> None:
    catalog = _catalog()
    save_frozen_manifest(
        tmp_path,
        queries=pd.DataFrame({"query_id": [1], "query": ["movie"], "query_type": ["genre_tag"]}),
        labels=pd.DataFrame({"query_id": [1], "movie_id": [1], "label": [3]}),
        user_context=pd.DataFrame({"user_id": [0]}),
        train_query_ids=[1], val_query_ids=[], test_query_ids=[],
        catalog=catalog, seed=42, data_source="fixture",
    )
    changed = catalog.copy()
    changed.loc[0, "clean_title"] = "Changed"
    try:
        load_frozen_manifest(tmp_path, catalog=changed)
    except ValueError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("catalog drift should fail")
