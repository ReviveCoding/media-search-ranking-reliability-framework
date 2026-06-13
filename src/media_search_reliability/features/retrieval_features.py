from __future__ import annotations

import math
import numpy as np
import pandas as pd

from media_search_reliability.text_normalization import canonical_genre, canonical_tag, split_pipe


def _split_pipe_set(x: object, *, kind: str = "tag") -> set[str]:
    return set(split_pipe(x, kind=kind))


def _split_genres_any(x: object) -> set[str]:
    text = str(x or "").replace(",", "|")
    return _split_pipe_set(text, kind="genre")


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def build_candidate_features(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    labels: pd.DataFrame,
    user_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = (
        candidates.merge(queries, on="query_id", how="left")
        .merge(catalog, on="movie_id", how="left")
        .merge(labels, on=["query_id", "movie_id"], how="left")
    )
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if user_context is not None and len(user_context):
        df = df.merge(user_context, on="user_id", how="left")
    else:
        df["preferred_genres"] = ""
        df["preferred_tags"] = ""

    genre_sets = df["genres"].map(lambda x: _split_pipe_set(x, kind="genre"))
    tag_sets = df["tag_text"].map(lambda x: _split_pipe_set(x, kind="tag"))
    tag_genome_sets = df.get("tag_genome_text", pd.Series("", index=df.index)).fillna("").map(
        lambda x: _split_pipe_set(x, kind="tag")
    )
    imdb_genre_sets = df.get("imdb_genres", pd.Series("", index=df.index)).fillna("").map(_split_genres_any)
    preferred_genre_sets = df["preferred_genres"].fillna("").map(lambda x: _split_pipe_set(x, kind="genre"))
    preferred_tag_sets = df["preferred_tags"].fillna("").map(lambda x: _split_pipe_set(x, kind="tag"))
    target_genres = df["target_genre"].fillna("").map(canonical_genre)
    target_tags = df["target_tag"].fillna("").map(canonical_tag)
    target_mood_tags = df.get("target_mood_tag", pd.Series("", index=df.index)).fillna("").map(canonical_tag)

    df["genre_overlap"] = [int(bool(target) and target in values) for target, values in zip(target_genres, genre_sets)]
    df["tag_overlap"] = [int(bool(target) and target in values) for target, values in zip(target_tags, tag_sets)]
    df["mood_overlap"] = [int(bool(target) and target in values) for target, values in zip(target_mood_tags, tag_sets)]
    df["tag_genome_tag_overlap"] = [
        int(bool(target) and target in values) for target, values in zip(target_tags, tag_genome_sets)
    ]
    df["tag_genome_mood_overlap"] = [
        int(bool(target) and target in values) for target, values in zip(target_mood_tags, tag_genome_sets)
    ]
    df["imdb_genre_overlap"] = [
        int(bool(target) and target in values) for target, values in zip(target_genres, imdb_genre_sets)
    ]
    df["user_genre_affinity"] = [
        len(values & prefs) / max(1, len(prefs)) for values, prefs in zip(genre_sets, preferred_genre_sets)
    ]
    df["user_tag_affinity"] = [
        len(values & prefs) / max(1, len(prefs)) for values, prefs in zip(tag_sets, preferred_tag_sets)
    ]
    df["tag_genome_user_affinity"] = [
        len(values & prefs) / max(1, len(prefs)) for values, prefs in zip(tag_genome_sets, preferred_tag_sets)
    ]
    df["user_history_confidence"] = [
        min(1.0, (len(g) + len(t)) / 6.0) for g, t in zip(preferred_genre_sets, preferred_tag_sets)
    ]

    target_decade = pd.to_numeric(df.get("target_decade", -1), errors="coerce").fillna(-1).astype(int)
    year = pd.to_numeric(df["year"], errors="coerce")
    movie_decade = (year.fillna(-1).astype(int) // 10) * 10
    df["decade_match"] = ((target_decade >= 0) & (movie_decade == target_decade)).astype(int)
    df["anchor_match"] = (
        pd.to_numeric(df.get("anchor_movie_id", -1), errors="coerce").fillna(-1).astype(int)
        == pd.to_numeric(df["movie_id"], errors="coerce").fillna(-2).astype(int)
    ).astype(int)

    anchor_columns = ["movie_id", "genres", "tag_text", "year", "rating_mean", "rating_count"]
    for optional in ("tag_genome_text", "imdb_genres"):
        if optional in catalog.columns:
            anchor_columns.append(optional)
    anchor_meta = catalog[anchor_columns].rename(columns={
        "movie_id": "anchor_movie_id",
        "genres": "anchor_genres",
        "tag_text": "anchor_tags",
        "tag_genome_text": "anchor_tag_genome_text",
        "imdb_genres": "anchor_imdb_genres",
        "year": "anchor_year",
        "rating_mean": "anchor_rating_mean",
        "rating_count": "anchor_rating_count",
    })
    df = df.merge(anchor_meta, on="anchor_movie_id", how="left")
    anchor_genre_sets = df["anchor_genres"].fillna("").map(lambda x: _split_pipe_set(x, kind="genre"))
    anchor_tag_sets = df["anchor_tags"].fillna("").map(lambda x: _split_pipe_set(x, kind="tag"))
    anchor_tag_genome_sets = df.get("anchor_tag_genome_text", pd.Series("", index=df.index)).fillna("").map(
        lambda x: _split_pipe_set(x, kind="tag")
    )
    df["anchor_genre_jaccard"] = [_jaccard(a, b) for a, b in zip(anchor_genre_sets, genre_sets)]
    df["anchor_tag_jaccard"] = [_jaccard(a, b) for a, b in zip(anchor_tag_sets, tag_sets)]
    df["anchor_tag_genome_jaccard"] = [
        _jaccard(a, b) for a, b in zip(anchor_tag_genome_sets, tag_genome_sets)
    ]
    anchor_year = pd.to_numeric(df["anchor_year"], errors="coerce")
    df["anchor_year_similarity"] = np.exp(-np.abs(year.fillna(2000) - anchor_year.fillna(2000)) / 15.0)
    anchor_rating = pd.to_numeric(df["anchor_rating_mean"], errors="coerce").fillna(3.0)
    candidate_rating = pd.to_numeric(df["rating_mean"], errors="coerce").fillna(3.0)
    df["anchor_rating_similarity"] = np.exp(-np.abs(candidate_rating - anchor_rating))

    query_type = df.get("query_type", pd.Series("adhoc", index=df.index)).fillna("adhoc").astype(str)
    df["query_is_similar_to"] = (query_type == "similar_to").astype(int)
    df["query_is_personalized"] = (query_type == "personalized").astype(int)
    df["query_is_visual"] = (query_type == "visual_query").astype(int)
    df["query_is_mood_decade"] = (query_type == "mood_decade").astype(int)
    df["movie_popularity_log"] = df["rating_count"].fillna(0).map(lambda value: math.log(float(value) + 1.0))
    median_year = year.median() if year.notna().any() else 2000
    df["movie_recency"] = year.fillna(median_year) - 1980
    df["rating_mean"] = candidate_rating
    df["tag_genome_coverage"] = pd.to_numeric(
        df.get("tag_genome_coverage", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    df["imdb_rating"] = pd.to_numeric(df.get("imdb_rating", candidate_rating), errors="coerce").fillna(candidate_rating)
    df["imdb_rating_centered"] = (df["imdb_rating"] - 5.0) / 5.0
    df["imdb_votes_log"] = pd.to_numeric(
        df.get("imdb_votes_log", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    max_votes_log = max(1.0, float(df["imdb_votes_log"].max()))
    df["imdb_vote_confidence"] = df["imdb_votes_log"] / max_votes_log
    runtime = pd.to_numeric(
        df.get("imdb_runtime_minutes", pd.Series(float("nan"), index=df.index)), errors="coerce"
    )
    runtime_median = runtime.median() if runtime.notna().any() else 100.0
    df["imdb_runtime_scaled"] = runtime.fillna(runtime_median) / 180.0
    df["imdb_coverage"] = pd.to_numeric(
        df.get("imdb_coverage", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    df["long_tail_flag"] = df["long_tail_flag"].fillna(0).astype(int)
    df["cold_start_flag"] = df["cold_start_flag"].fillna(0).astype(int)
    for column in (
        "bm25_score", "dense_score", "hybrid_score", "anchor_dense_score", "anchor_metadata_score",
        "personalized_dense_score", "specialized_score",
    ):
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


FEATURE_COLUMNS = [
    "bm25_score", "dense_score", "hybrid_score", "specialized_score",
    "anchor_dense_score", "anchor_metadata_score", "personalized_dense_score",
    "genre_overlap", "tag_overlap", "mood_overlap",
    "tag_genome_tag_overlap", "tag_genome_mood_overlap", "imdb_genre_overlap",
    "user_genre_affinity", "user_tag_affinity", "tag_genome_user_affinity", "user_history_confidence",
    "anchor_genre_jaccard", "anchor_tag_jaccard", "anchor_tag_genome_jaccard",
    "anchor_year_similarity", "anchor_rating_similarity",
    "movie_popularity_log", "movie_recency", "rating_mean", "tag_genome_coverage",
    "imdb_rating", "imdb_rating_centered", "imdb_votes_log", "imdb_vote_confidence",
    "imdb_runtime_scaled", "imdb_coverage",
    "long_tail_flag", "cold_start_flag",
    "decade_match", "anchor_match", "query_is_similar_to", "query_is_personalized", "query_is_visual",
    "query_is_mood_decade",
]
