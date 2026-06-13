from __future__ import annotations

import pandas as pd

from media_search_reliability.data_ingestion.enrichment import apply_optional_enrichment
from media_search_reliability.features.retrieval_features import build_candidate_features


def test_feature_only_enrichment_does_not_rewrite_retrieval_text(tmp_path) -> None:
    catalog = pd.DataFrame({
        "movie_id": [1, 2],
        "clean_title": ["Robot Film", "Drama Film"],
        "genres": ["Sci-Fi", "Drama"],
        "tag_text": ["robot|space", "emotional"],
        "content_text": ["Robot Film Sci-Fi robot space", "Drama Film Drama emotional"],
        "year": [2000, 2001],
        "rating_mean": [4.0, 3.5],
        "rating_count": [100, 50],
        "long_tail_flag": [0, 1],
        "cold_start_flag": [0, 0],
    })
    tag_path = tmp_path / "tag.csv"
    pd.DataFrame({
        "movie_id": [1],
        "tag_genome_text": ["artificial intelligence|cyborg"],
    }).to_csv(tag_path, index=False)
    enriched, diagnostics = apply_optional_enrichment(catalog, {
        "mode": "feature_only",
        "tag_genome_enrichment_path": str(tag_path),
    })
    assert enriched["content_text"].tolist() == catalog["content_text"].tolist()
    assert enriched["tag_text"].tolist() == catalog["tag_text"].tolist()
    assert diagnostics["retrieval_text_changed"] is False
    assert enriched.loc[0, "tag_genome_coverage"] == 1


def test_feature_only_metadata_creates_numeric_ranking_features(tmp_path) -> None:
    catalog = pd.DataFrame({
        "movie_id": [1], "clean_title": ["Robot Film"], "genres": ["Sci-Fi"],
        "tag_text": ["robot"], "content_text": ["Robot Film Sci-Fi robot"],
        "year": [2000], "rating_mean": [4.0], "rating_count": [100],
        "long_tail_flag": [0], "cold_start_flag": [0],
        "tag_genome_text": ["robot|funny"], "tag_genome_coverage": [1],
        "imdb_rating": [7.5], "imdb_votes_log": [10.0], "imdb_runtime_minutes": [100],
        "imdb_coverage": [1], "imdb_genres": ["Sci-Fi,Comedy"],
    })
    queries = pd.DataFrame({
        "query_id": [1], "query": ["funny sci-fi"], "query_type": ["genre_tag"],
        "user_id": [0], "target_genre": ["Sci-Fi"], "target_tag": ["robot"],
        "target_mood_tag": ["funny"], "target_decade": [2000], "anchor_movie_id": [-1],
    })
    candidates = pd.DataFrame({
        "query_id": [1], "movie_id": [1], "bm25_score": [1.0], "dense_score": [1.0],
        "hybrid_score": [1.0], "specialized_score": [0.0], "anchor_dense_score": [0.0],
        "anchor_metadata_score": [0.0], "personalized_dense_score": [0.0],
    })
    labels = pd.DataFrame({"query_id": [1], "movie_id": [1], "label": [3]})
    users = pd.DataFrame({"user_id": [0], "preferred_genres": ["Sci-Fi"], "preferred_tags": ["robot"]})
    features = build_candidate_features(candidates, queries, catalog, labels, users)
    row = features.iloc[0]
    assert row["tag_genome_tag_overlap"] == 1
    assert row["tag_genome_mood_overlap"] == 1
    assert row["imdb_genre_overlap"] == 1
    assert row["imdb_vote_confidence"] > 0
