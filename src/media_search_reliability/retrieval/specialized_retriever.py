from __future__ import annotations

import numpy as np
import pandas as pd

from media_search_reliability.text_normalization import split_pipe
from media_search_reliability.utils import normalize_scores


class SpecializedCandidateGenerator:
    """Query-type-aware candidate generation for similar-to and personalized search."""

    def __init__(self, catalog: pd.DataFrame, dense, vector_index, user_context: pd.DataFrame | None = None):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.dense = dense
        self.vector_index = vector_index
        self.movie_to_pos = {int(mid): pos for pos, mid in enumerate(self.catalog["movie_id"].astype(int).tolist())}
        self.catalog_by_id = self.catalog.set_index(self.catalog["movie_id"].astype(int), drop=False)
        self.user_lookup = {}
        if user_context is not None and len(user_context):
            self.user_lookup = user_context.set_index("user_id").to_dict("index")

    @staticmethod
    def _to_score_map(rows: list[tuple[int, float]]) -> dict[int, float]:
        if not rows:
            return {}
        ids = [int(mid) for mid, _ in rows]
        values = normalize_scores(np.asarray([float(score) for _, score in rows]))
        return {mid: float(score) for mid, score in zip(ids, values)}

    def _metadata_neighbors(self, anchor_id: int, top_k: int) -> dict[int, float]:
        if anchor_id not in self.catalog_by_id.index:
            return {}
        anchor = self.catalog_by_id.loc[anchor_id]
        anchor_genres = set(split_pipe(anchor.get("genres", ""), kind="genre"))
        anchor_tags = set(split_pipe(anchor.get("tag_text", ""), kind="tag"))
        rows = []
        for row in self.catalog.itertuples(index=False):
            movie_id = int(row.movie_id)
            if movie_id == anchor_id:
                continue
            genres = set(split_pipe(getattr(row, "genres", ""), kind="genre"))
            tags = set(split_pipe(getattr(row, "tag_text", ""), kind="tag"))
            genre_score = len(anchor_genres & genres) / max(1, len(anchor_genres | genres))
            tag_score = len(anchor_tags & tags) / max(1, len(anchor_tags | tags))
            score = 0.6 * genre_score + 0.4 * tag_score
            if score > 0:
                rows.append((movie_id, score))
        rows.sort(key=lambda x: (-x[1], x[0]))
        return self._to_score_map(rows[:top_k])

    def similar_to(self, anchor_id: int, top_k: int) -> tuple[dict[int, float], dict[int, float]]:
        pos = self.movie_to_pos.get(int(anchor_id), None)
        dense_map: dict[int, float] = {}
        if pos is not None:
            query_embedding = np.asarray(self.dense.embeddings[pos], dtype="float32").reshape(1, -1)
            rows = self.vector_index.search(query_embedding, top_k=min(len(self.catalog), max(top_k * 3, 100)))[0]
            rows = [(int(mid), float(score)) for mid, score in rows if int(mid) != int(anchor_id)]
            dense_map = self._to_score_map(rows[:top_k])
        return dense_map, self._metadata_neighbors(int(anchor_id), top_k)

    def personalized(self, user_id: int, target_genre: str, target_tag: str, top_k: int, preferred_genres: str = "", preferred_tags: str = "") -> dict[int, float]:
        profile = self.user_lookup.get(int(user_id), {})
        genres = str(preferred_genres or profile.get("preferred_genres", "")).replace("|", " ")
        tags = str(preferred_tags or profile.get("preferred_tags", "")).replace("|", " ")
        profile_text = " ".join(part for part in [genres, tags, str(target_genre), str(target_tag)] if part).strip()
        if not profile_text:
            return {}
        q_emb = self.dense.encode_queries([profile_text])
        rows = self.vector_index.search(q_emb, top_k=min(len(self.catalog), max(top_k * 3, 100)))[0]
        return self._to_score_map([(int(mid), float(score)) for mid, score in rows[:top_k]])

    def retrieve(self, query_row, top_k: int) -> dict[str, dict[int, float]]:
        qtype = str(getattr(query_row, "query_type", ""))
        anchor_dense: dict[int, float] = {}
        anchor_metadata: dict[int, float] = {}
        personalized_dense: dict[int, float] = {}
        if qtype == "similar_to":
            anchor_dense, anchor_metadata = self.similar_to(int(getattr(query_row, "anchor_movie_id", -1)), top_k)
        elif qtype == "personalized":
            personalized_dense = self.personalized(
                int(getattr(query_row, "user_id", -1)),
                str(getattr(query_row, "target_genre", "")),
                str(getattr(query_row, "target_tag", "")),
                top_k,
                str(getattr(query_row, "preferred_genres", "")),
                str(getattr(query_row, "preferred_tags", "")),
            )
        return {
            "anchor_dense_score": anchor_dense,
            "anchor_metadata_score": anchor_metadata,
            "personalized_dense_score": personalized_dense,
        }
