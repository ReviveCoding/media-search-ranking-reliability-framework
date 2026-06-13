from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FrozenBenchmarkManifest:
    queries: pd.DataFrame
    labels: pd.DataFrame
    user_context: pd.DataFrame
    train_query_ids: list[int]
    val_query_ids: list[int]
    test_query_ids: list[int]
    metadata: dict


def catalog_fingerprint(catalog: pd.DataFrame) -> str:
    required = [c for c in ("movie_id", "clean_title", "year") if c in catalog.columns]
    if "movie_id" not in required:
        raise ValueError("Catalog must contain movie_id for fingerprinting")
    stable = catalog[required].copy().sort_values("movie_id", kind="mergesort")
    payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_frozen_manifest(
    manifest_dir: str | Path,
    *,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    user_context: pd.DataFrame,
    train_query_ids: list[int],
    val_query_ids: list[int],
    test_query_ids: list[int],
    catalog: pd.DataFrame,
    seed: int,
    data_source: str,
) -> Path:
    root = Path(manifest_dir)
    root.mkdir(parents=True, exist_ok=True)
    queries.to_csv(root / "queries.csv", index=False)
    labels.to_csv(root / "judgments.csv", index=False)
    user_context.to_csv(root / "user_context.csv", index=False)
    splits = {
        "train_query_ids": [int(x) for x in train_query_ids],
        "val_query_ids": [int(x) for x in val_query_ids],
        "test_query_ids": [int(x) for x in test_query_ids],
    }
    (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    metadata = {
        "version": 1,
        "seed": int(seed),
        "data_source": str(data_source),
        "catalog_fingerprint": catalog_fingerprint(catalog),
        "catalog_size": int(len(catalog)),
        "query_count": int(len(queries)),
        "judgment_count": int(len(labels)),
    }
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return root


def load_frozen_manifest(
    manifest_dir: str | Path,
    *,
    catalog: pd.DataFrame,
    require_catalog_fingerprint: bool = True,
) -> FrozenBenchmarkManifest:
    root = Path(manifest_dir)
    required = ["queries.csv", "judgments.csv", "user_context.csv", "splits.json", "manifest.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Frozen benchmark manifest is incomplete: {missing}")

    queries = pd.read_csv(root / "queries.csv")
    labels = pd.read_csv(root / "judgments.csv")
    user_context = pd.read_csv(root / "user_context.csv")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    current_fingerprint = catalog_fingerprint(catalog)
    expected_fingerprint = str(metadata.get("catalog_fingerprint", ""))
    if require_catalog_fingerprint and current_fingerprint != expected_fingerprint:
        raise ValueError(
            "Catalog fingerprint differs from the frozen benchmark manifest. "
            f"expected={expected_fingerprint}, current={current_fingerprint}"
        )

    valid_movie_ids = set(catalog["movie_id"].astype(int))
    unknown = set(labels["movie_id"].astype(int)) - valid_movie_ids
    if unknown:
        raise ValueError(f"Frozen judgments reference {len(unknown)} movie ids absent from the catalog")

    return FrozenBenchmarkManifest(
        queries=queries,
        labels=labels,
        user_context=user_context,
        train_query_ids=[int(x) for x in splits["train_query_ids"]],
        val_query_ids=[int(x) for x in splits["val_query_ids"]],
        test_query_ids=[int(x) for x in splits["test_query_ids"]],
        metadata=metadata,
    )
