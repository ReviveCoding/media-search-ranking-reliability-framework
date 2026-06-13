from __future__ import annotations

from pathlib import Path
import pandas as pd


def validate_catalog(catalog: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame | None = None) -> dict:
    stats = {
        "num_movies": int(len(catalog)),
        "num_ratings": int(len(ratings)),
        "num_users": int(ratings["user_id"].nunique()) if len(ratings) else 0,
        "num_tags": int(len(tags)) if tags is not None else 0,
        "missing_title": int(catalog["title"].isna().sum()) if "title" in catalog else 0,
        "missing_genres": int(catalog["genres"].isna().sum()) if "genres" in catalog else 0,
        "duplicate_movie_id": int(catalog["movie_id"].duplicated().sum()) if "movie_id" in catalog else 0,
        "cold_start_movies": int(catalog.get("cold_start_flag", pd.Series([0]*len(catalog))).sum()),
        "long_tail_movies": int(catalog.get("long_tail_flag", pd.Series([0]*len(catalog))).sum()),
        "tag_coverage_rate": float((catalog.get("tag_text", pd.Series([""]*len(catalog))).fillna("").str.len() > 0).mean()) if len(catalog) else 0.0,
        "rating_min": float(ratings["rating"].min()) if len(ratings) else None,
        "rating_max": float(ratings["rating"].max()) if len(ratings) else None,
    }
    return stats


def write_validation_report(stats: dict, path: str | Path) -> None:
    lines = ["# Data Validation Report", ""]
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
