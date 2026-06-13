from __future__ import annotations

import numpy as np
import pandas as pd

from media_search_reliability.query_understanding import QueryUnderstanding
from media_search_reliability.retrieval.faiss_index import VectorIndex
from media_search_reliability.retrieval.specialized_retriever import SpecializedCandidateGenerator
from media_search_reliability.text_normalization import canonical_genre, canonical_tag


class _Dense:
    def __init__(self):
        self.embeddings = np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype="float32")

    def encode_queries(self, queries):
        return np.asarray([[1.0, 0.0] for _ in queries], dtype="float32")


def _catalog():
    return pd.DataFrame({
        "movie_id": [1, 2, 3],
        "clean_title": ["Hidden Planet", "Robot World", "Love Story"],
        "genres": ["Sci-Fi|Comedy", "Sci-Fi", "Romance"],
        "tag_text": ["robots|funny!|space", "robot|space", "romance"],
        "year": [2000, 2002, 2001],
        "rating_mean": [4.0, 3.9, 4.1],
        "rating_count": [100, 80, 120],
    })


def test_shared_query_normalization():
    assert canonical_tag("robots") == "robot"
    assert canonical_tag("funny!") == "funny"
    assert canonical_genre("science fiction") == "sci-fi"
    parsed = QueryUnderstanding(_catalog()).parse("funny! science fiction movie with robots")
    assert parsed.target_genre == "Sci-Fi"
    assert parsed.target_tag == "robot"
    assert parsed.target_mood_tag == "funny"


def test_specialized_similar_to_excludes_anchor_and_finds_neighbor():
    dense = _Dense()
    index = VectorIndex(use_gpu_if_available=False).fit(dense.embeddings, [1, 2, 3])
    generator = SpecializedCandidateGenerator(_catalog(), dense, index)
    dense_scores, metadata_scores = generator.similar_to(1, top_k=3)
    assert 1 not in dense_scores
    assert 1 not in metadata_scores
    assert 2 in dense_scores
    assert 2 in metadata_scores


def test_personalized_profile_retrieval_uses_preferences():
    dense = _Dense()
    index = VectorIndex(use_gpu_if_available=False).fit(dense.embeddings, [1, 2, 3])
    users = pd.DataFrame({"user_id": [7], "preferred_genres": ["Sci-Fi|Comedy"], "preferred_tags": ["robot|space"]})
    generator = SpecializedCandidateGenerator(_catalog(), dense, index, users)
    scores = generator.personalized(7, "Sci-Fi", "robot", top_k=3)
    assert scores
    assert 1 in scores or 2 in scores
