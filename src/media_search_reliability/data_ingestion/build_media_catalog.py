from __future__ import annotations

import re
import pandas as pd

from media_search_reliability.text_normalization import canonical_tag


def extract_year(title: str):
    m = re.search(r"\((\d{4})\)", str(title))
    return int(m.group(1)) if m else None


def clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*", "", str(title)).strip()


def build_media_catalog(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["year"] = movies["title"].apply(extract_year)
    movies["clean_title"] = movies["title"].apply(clean_title)
    movies["genres"] = movies["genres"].fillna("Unknown")

    rating_stats = ratings.groupby("movie_id").agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
    ).reset_index()
    rating_stats["rating_std"] = rating_stats["rating_std"].fillna(0.0)

    if len(tags):
        normalized_tags = tags.dropna(subset=["tag"])[["movie_id", "tag"]].copy()
        normalized_tags["tag"] = normalized_tags["tag"].astype(str).map(canonical_tag)
        normalized_tags = normalized_tags[normalized_tags["tag"].astype(str).str.len() > 0]
        tag_counts = (
            normalized_tags.groupby(["movie_id", "tag"], as_index=False)
            .size()
            .sort_values(["movie_id", "size", "tag"], ascending=[True, False, True], kind="mergesort")
        )
        tag_agg = (
            tag_counts.groupby("movie_id", sort=True)["tag"]
            .apply(lambda values: "|".join(values.head(8)))
            .reset_index(name="tag_text")
        )
    else:
        tag_agg = pd.DataFrame(columns=["movie_id", "tag_text"])
    if "tag" in tag_agg.columns:
        tag_agg = tag_agg.rename(columns={"tag": "tag_text"})

    catalog = movies.merge(rating_stats, on="movie_id", how="left").merge(tag_agg, on="movie_id", how="left")
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype(int)
    catalog["rating_mean"] = catalog["rating_mean"].fillna(catalog["rating_mean"].mean() if catalog["rating_mean"].notna().any() else 3.0)
    catalog["tag_text"] = catalog["tag_text"].fillna("")
    catalog["long_tail_flag"] = (catalog["rating_count"] <= catalog["rating_count"].quantile(0.25)).astype(int)
    catalog["cold_start_flag"] = (catalog["rating_count"] <= 5).astype(int)
    catalog["popularity_bucket"] = pd.qcut(catalog["rating_count"].rank(method="first"), q=5, labels=False, duplicates="drop").fillna(0).astype(int)
    catalog["content_text"] = (
        catalog["clean_title"].astype(str) + " " +
        catalog["genres"].astype(str).str.replace("|", " ", regex=False) + " " +
        catalog["tag_text"].astype(str).str.replace("|", " ", regex=False)
    )
    catalog["synthetic_description"] = catalog.apply(
        lambda r: f"{r.clean_title} is a {str(r.genres).replace('|', ', ')} movie with tags {str(r.tag_text).replace('|', ', ')}.", axis=1
    )
    return catalog
