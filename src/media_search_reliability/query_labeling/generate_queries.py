from __future__ import annotations

import numpy as np
import pandas as pd

from media_search_reliability.query_semantics import MOODS, MOOD_TO_TAG
from media_search_reliability.text_normalization import canonical_genre, canonical_tag, split_pipe as canonical_split_pipe
QUERY_BLUEPRINTS = [
    ("genre_tag", "{genre} movie with {tag}"),
    ("similar_to", "movie like {title}"),
    ("personalized", "recommend {genre} movies for a user who likes {pref}"),
    ("mood_decade", "{mood} movie from the {decade}s"),
    ("visual_query", "movie with scenes of {tag}"),
]


def _rng(seed: int) -> np.random.Generator:
    bitgen = np.random.PCG64DXSM(seed) if hasattr(np.random, "PCG64DXSM") else np.random.PCG64(seed)
    return np.random.Generator(bitgen)


def _split_pipe(x: str, kind: str = "tag") -> list[str]:
    return canonical_split_pipe(x, kind=kind)


def build_user_context(ratings: pd.DataFrame, catalog: pd.DataFrame, max_users: int = 1000) -> pd.DataFrame:
    """Build observed user preferences from noisy interaction history."""
    joined = ratings.merge(catalog[["movie_id", "genres", "tag_text"]], on="movie_id", how="left")
    rows = []
    for user_id, grp in joined.groupby("user_id", sort=True):
        if len(rows) >= max_users:
            break
        positive = grp[grp["rating"] >= 4.0]
        source = positive if len(positive) else grp
        genres, tags = [], []
        for _, row in source.sort_values(["rating", "movie_id"], ascending=[False, True]).head(80).iterrows():
            genres.extend(_split_pipe(row.get("genres", ""), kind="genre"))
            tags.extend(_split_pipe(row.get("tag_text", ""), kind="tag"))
        top_genres = pd.Series(genres).value_counts().sort_index().sort_values(ascending=False, kind="stable").index[:3].tolist() if genres else []
        top_tags = pd.Series(tags).value_counts().sort_index().sort_values(ascending=False, kind="stable").index[:3].tolist() if tags else []
        rows.append({
            "user_id": int(user_id),
            "preferred_genres": "|".join(top_genres),
            "preferred_tags": "|".join(top_tags),
        })
    return pd.DataFrame(rows)


def build_latent_user_context(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return simulator-known user preferences, unavailable in MovieLens mode."""
    if "latent_preferred_genres" not in ratings.columns:
        return pd.DataFrame(columns=["user_id", "preferred_genres", "preferred_tags"])
    frame = ratings[["user_id", "latent_preferred_genres"]].drop_duplicates("user_id").sort_values("user_id")
    return frame.rename(columns={"latent_preferred_genres": "preferred_genres"}).assign(preferred_tags="")


def _movie_decade(year) -> int:
    if pd.isna(year):
        return 2000
    return int(int(year) // 10 * 10)


def _candidate_tags(candidate: pd.Series, latent: bool = True) -> set[str]:
    """Return latent simulator tags or observed catalog tags.

    In public-data mode no latent tags exist, so both paths use observed tags.
    """
    latent_tags = set(_split_pipe(candidate.get("synthetic_tags", ""), kind="tag"))
    observed_tags = set(_split_pipe(candidate.get("tag_text", ""), kind="tag"))
    return latent_tags if latent and latent_tags else observed_tags


def _relevance_label(
    candidate: pd.Series,
    anchor: pd.Series,
    qtype: str,
    anchor_genres: set[str],
    anchor_tags: set[str],
    target_genre: str,
    target_tag: str,
    target_mood_tag: str,
    target_decade: int,
    quality_threshold: float,
    preferred_genres: set[str],
    use_latent_signals: bool,
) -> int:
    candidate_genres = set(_split_pipe(candidate.get("genres", ""), kind="genre"))
    candidate_tags = _candidate_tags(candidate, latent=use_latent_signals)
    candidate_decade = _movie_decade(candidate.get("year"))
    genre_match = target_genre in candidate_genres
    tag_match = target_tag in candidate_tags
    mood_match = target_mood_tag in candidate_tags if target_mood_tag else False
    decade_match = candidate_decade == target_decade
    quality_value = candidate.get("latent_quality", candidate.get("rating_mean", 3.0)) if use_latent_signals else candidate.get("rating_mean", 3.0)
    quality_match = float(quality_value) >= quality_threshold
    preference_match = bool(preferred_genres.intersection(candidate_genres))
    is_anchor = int(candidate.movie_id) == int(anchor.movie_id)

    # "movie like X" should not return X itself. This removes a trivial exact-title
    # positive that previously inflated MRR and teaches the ranker to demote self matches.
    if qtype == "similar_to":
        if is_anchor:
            return 0
        semantic_overlap = len(anchor_genres.intersection(candidate_genres)) + len(anchor_tags.intersection(candidate_tags))
        if semantic_overlap >= 2 and quality_match:
            return 3
        if semantic_overlap >= 1 and quality_match:
            return 2
        if semantic_overlap >= 1:
            return 1
        return 0

    if qtype == "genre_tag":
        if genre_match and tag_match:
            return 3
        if (genre_match or tag_match) and quality_match:
            return 2
        return 1 if (genre_match or tag_match) else 0

    if qtype == "personalized":
        if genre_match and preference_match:
            return 3
        if (genre_match or preference_match) and quality_match:
            return 2
        return 1 if (genre_match or preference_match) else 0

    if qtype == "mood_decade":
        if mood_match and decade_match:
            return 3
        if (mood_match or decade_match) and (genre_match or quality_match):
            return 2
        return 1 if (mood_match or decade_match) else 0

    if qtype == "visual_query":
        if tag_match:
            return 3
        # A genre match is only a weak fallback because the visual intent is the scene tag.
        return 1 if genre_match else 0

    if genre_match and tag_match:
        return 3
    if genre_match or tag_match:
        return 2 if quality_match else 1
    return 0


def generate_queries_and_labels(
    catalog: pd.DataFrame,
    ratings: pd.DataFrame,
    num_queries: int = 450,
    candidates_per_query: int = 80,
    seed: int = 42,
    label_all_catalog: bool = False,
    label_noise: float = 0.0,
):
    """Generate stratified natural-language queries and 0/1/2/3 labels.

    For small Monte Carlo runs, ``label_all_catalog=True`` produces complete
    synthetic judgments over the catalog, preventing retrieval metrics from using
    only the retrieved subset as their recall denominator. Large MovieLens runs
    can retain pooled judging for memory efficiency.
    """
    if not 0 <= label_noise <= 1:
        raise ValueError("label_noise must be in [0, 1].")
    rng = _rng(seed)
    catalog = catalog.reset_index(drop=True).copy()
    user_ctx = build_user_context(ratings, catalog, max_users=max(50, num_queries // 2))
    if len(user_ctx) == 0:
        user_ctx = pd.DataFrame({"user_id": [0], "preferred_genres": ["Comedy|Drama"], "preferred_tags": [""]})
    latent_user_ctx = build_latent_user_context(ratings)
    query_user_ctx = latent_user_ctx if len(latent_user_ctx) else user_ctx
    observed_context_lookup = user_ctx.set_index("user_id").to_dict("index") if len(user_ctx) else {}

    queries, labels = [], []
    all_indices = np.arange(len(catalog))
    observed_quality_threshold = float(catalog["rating_mean"].median()) if catalog["rating_mean"].notna().any() else 3.5
    if "latent_quality" in catalog.columns and catalog["latent_quality"].notna().any():
        latent_quality_threshold = float(catalog["latent_quality"].median())
    else:
        latent_quality_threshold = observed_quality_threshold

    for qid in range(num_queries):
        anchor_idx = int(rng.integers(0, len(catalog)))
        anchor = catalog.iloc[anchor_idx]
        genres = _split_pipe(anchor.genres, kind="genre") or ["drama"]
        tags = _candidate_tags(anchor) or {str(rng.choice(["robots", "family", "detective", "space", "romance"]))}
        genre = canonical_genre(str(rng.choice(genres)))
        tag = canonical_tag(str(rng.choice(sorted(tags))))
        qtype, template = QUERY_BLUEPRINTS[qid % len(QUERY_BLUEPRINTS)]
        anchor_tags = set(tags)
        observed_anchor_tags = _candidate_tags(anchor, latent=False)
        compatible_moods = [m for m in MOODS if MOOD_TO_TAG[m] in anchor_tags]
        mood = str(rng.choice(compatible_moods if compatible_moods else MOODS))
        target_mood_tag = canonical_tag(MOOD_TO_TAG[mood])
        decade = _movie_decade(anchor.year)
        user = query_user_ctx.iloc[int(rng.integers(0, len(query_user_ctx)))]
        preferred_genres = set(_split_pipe(user.preferred_genres, kind="genre"))
        observed_profile = observed_context_lookup.get(int(user.user_id), {})
        observed_preferred_genres = set(_split_pipe(observed_profile.get("preferred_genres", ""), kind="genre"))
        pref = sorted(preferred_genres)[0] if preferred_genres else genre
        query = template.format(mood=mood, genre=genre, tag=tag, title=anchor.clean_title, decade=decade, pref=pref)
        queries.append({
            "query_id": qid,
            "query": query,
            "query_type": qtype,
            "anchor_movie_id": int(anchor.movie_id),
            "user_id": int(user.user_id),
            "target_genre": genre,
            "target_tag": tag,
            "target_mood_tag": target_mood_tag,
            "target_decade": int(decade),
        })

        if label_all_catalog:
            pool = all_indices
        else:
            genre_match = catalog[catalog["genres"].fillna("").str.contains(genre, regex=False)].index.values
            observed_tag_match = catalog[catalog["tag_text"].fillna("").str.contains(tag, regex=False)].index.values
            latent_tag_match = catalog[catalog.get("synthetic_tags", pd.Series("", index=catalog.index)).fillna("").str.contains(tag, regex=False)].index.values
            tag_match = np.unique(np.concatenate([observed_tag_match, latent_tag_match]))
            decade_match = catalog[catalog["year"].apply(_movie_decade) == decade].index.values
            popular = catalog.sort_values("rating_count", ascending=False).head(max(20, candidates_per_query // 4)).index.values
            random_size = min(len(all_indices), candidates_per_query * 3)
            pool = np.unique(np.concatenate([
                [anchor_idx], genre_match[:300], tag_match[:300], decade_match[:150], popular,
                rng.choice(all_indices, size=random_size, replace=False),
            ]))
            if len(pool) > candidates_per_query:
                must = np.unique(np.concatenate([[anchor_idx], genre_match[:8], tag_match[:8], decade_match[:5]])).astype(int)
                rest = np.setdiff1d(pool, must)
                remaining = max(0, candidates_per_query - len(must))
                chosen_rest = rng.choice(rest, size=min(len(rest), remaining), replace=False) if remaining and len(rest) else np.array([], dtype=int)
                pool = np.unique(np.concatenate([must, chosen_rest]))[:candidates_per_query]

        anchor_genres = set(genres)
        for idx in pool:
            candidate = catalog.iloc[int(idx)]
            clean_label = _relevance_label(
                candidate=candidate,
                anchor=anchor,
                qtype=qtype,
                anchor_genres=anchor_genres,
                anchor_tags=anchor_tags,
                target_genre=genre,
                target_tag=tag,
                target_mood_tag=target_mood_tag,
                target_decade=decade,
                quality_threshold=latent_quality_threshold,
                preferred_genres=preferred_genres,
                use_latent_signals=True,
            )
            observed_base_label = _relevance_label(
                candidate=candidate,
                anchor=anchor,
                qtype=qtype,
                anchor_genres=anchor_genres,
                anchor_tags=observed_anchor_tags,
                target_genre=genre,
                target_tag=tag,
                target_mood_tag=target_mood_tag,
                target_decade=decade,
                quality_threshold=observed_quality_threshold,
                preferred_genres=observed_preferred_genres,
                use_latent_signals=False,
            )
            observed_label = observed_base_label
            if label_noise > 0 and rng.random() < label_noise:
                observed_label = int(np.clip(observed_base_label + rng.choice([-1, 1]), 0, 3))
            labels.append({
                "query_id": qid,
                "movie_id": int(candidate.movie_id),
                "label": observed_label,
                "observed_base_label": observed_base_label,
                "clean_label": clean_label,
                "slice_long_tail": int(candidate.long_tail_flag),
                "slice_cold_start": int(candidate.cold_start_flag),
                "slice_visual_query": int(qtype == "visual_query"),
            })
    return pd.DataFrame(queries), pd.DataFrame(labels), user_ctx
