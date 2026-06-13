from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Iterable

import pandas as pd

from media_search_reliability.query_semantics import MOOD_TO_TAG
from media_search_reliability.text_normalization import (
    canonical_genre,
    canonical_genre_phrase,
    canonical_join,
    canonical_phrase,
    canonical_tag,
    split_pipe,
)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = canonical_phrase(phrase)
    return bool(normalized_phrase) and re.search(rf"\b{re.escape(normalized_phrase)}\b", text) is not None


@dataclass(frozen=True)
class ParsedQueryIntent:
    query_type: str = "adhoc"
    anchor_movie_id: int = -1
    target_genre: str = ""
    target_tag: str = ""
    target_mood_tag: str = ""
    target_decade: int = -1
    preferred_genres: str = ""
    preferred_tags: str = ""
    parser_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class QueryUnderstanding:
    """Deterministic parser with shared canonicalization for train-serving parity."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.copy()
        genre_display: dict[str, str] = {}
        tags: set[str] = set()
        for value in self.catalog.get("genres", pd.Series(dtype=str)).fillna(""):
            for raw in str(value).split("|"):
                key = canonical_genre(raw)
                if key and key not in genre_display:
                    genre_display[key] = str(raw).strip()
        for value in self.catalog.get("tag_text", pd.Series(dtype=str)).fillna(""):
            tags.update(split_pipe(value, kind="tag"))
        # Known lexical intents remain parseable even when a sparse catalog lacks the exact tag.
        tags.update({"robot", "funny", "family", "space", "detective", "psychological", "mystery", "heist"})
        self.genre_display = genre_display
        self.genres = sorted(genre_display, key=lambda value: (-len(value), value))
        self.tags = sorted(tags, key=lambda value: (-len(value), value))
        self.title_lookup: dict[str, int] = {}
        for row in self.catalog[["movie_id", "clean_title"]].dropna().itertuples(index=False):
            normalized = canonical_phrase(row.clean_title)
            if normalized and normalized not in self.title_lookup:
                self.title_lookup[normalized] = int(row.movie_id)

    def _resolve_anchor(self, query: str, anchor_movie_id: int | None) -> int:
        if anchor_movie_id is not None:
            return int(anchor_movie_id)
        normalized = canonical_phrase(query)
        match = re.search(r"\bmovie like\s+(.+)$", normalized)
        if not match:
            return -1
        candidate = match.group(1).strip()
        if candidate in self.title_lookup:
            return self.title_lookup[candidate]
        matches = [(title, movie_id) for title, movie_id in self.title_lookup.items() if title and title in candidate]
        return max(matches, key=lambda pair: len(pair[0]))[1] if matches else -1

    def _anchor_metadata(self, anchor_movie_id: int) -> tuple[str, str]:
        if anchor_movie_id < 0:
            return "", ""
        rows = self.catalog[self.catalog["movie_id"].astype(int) == int(anchor_movie_id)]
        if rows.empty:
            return "", ""
        row = rows.iloc[0]
        genres = split_pipe(row.get("genres", ""), kind="genre")
        tags = split_pipe(row.get("tag_text", ""), kind="tag")
        genre = self.genre_display.get(genres[0], genres[0]) if genres else ""
        return genre, (tags[0] if tags else "")

    def parse(
        self,
        query: str,
        anchor_movie_id: int | None = None,
        preferred_genres: Iterable[str] | None = None,
        preferred_tags: Iterable[str] | None = None,
    ) -> ParsedQueryIntent:
        normalized = canonical_phrase(query)
        anchor_id = self._resolve_anchor(query, anchor_movie_id)

        if "movie like" in normalized:
            query_type = "similar_to"
        elif "with scenes of" in normalized or "scenes of" in normalized:
            query_type = "visual_query"
        elif normalized.startswith("recommend ") or "for a user who likes" in normalized:
            query_type = "personalized"
        else:
            query_type = "adhoc"

        genre_normalized = canonical_genre_phrase(query)
        genre_key = next((genre for genre in self.genres if re.search(rf"\b{re.escape(genre.replace(chr(45), chr(32)))}\b", genre_normalized)), "")
        target_genre = self.genre_display.get(genre_key, genre_key)
        mood_word = next((mood for mood in MOOD_TO_TAG if _contains_phrase(normalized, mood)), "")
        target_mood_tag = canonical_tag(MOOD_TO_TAG.get(mood_word, ""))
        matched_tags = [canonical_tag(tag) for tag in self.tags if _contains_phrase(normalized, tag)]
        target_tag = next((tag for tag in matched_tags if tag and tag != target_mood_tag), matched_tags[0] if matched_tags else "")
        decade_match = re.search(r"\b((?:19|20)\d0)s\b", normalized)
        target_decade = int(decade_match.group(1)) if decade_match else -1

        if query_type == "adhoc":
            if target_mood_tag and target_decade >= 0:
                query_type = "mood_decade"
            elif target_genre or target_tag:
                query_type = "genre_tag"

        anchor_genre, anchor_tag = self._anchor_metadata(anchor_id)
        if query_type == "similar_to":
            target_genre = target_genre or anchor_genre
            target_tag = target_tag or anchor_tag

        preferred_genres_text = canonical_join(preferred_genres or [], kind="genre")
        preferred_tags_text = canonical_join(preferred_tags or [], kind="tag")

        signals = [
            bool(target_genre), bool(target_tag), bool(target_mood_tag),
            target_decade >= 0, anchor_id >= 0,
            bool(preferred_genres_text), bool(preferred_tags_text),
        ]
        confidence = min(1.0, 0.15 + 0.2 * sum(signals)) if query_type != "adhoc" else 0.1 * sum(signals)
        return ParsedQueryIntent(
            query_type=query_type,
            anchor_movie_id=anchor_id,
            target_genre=target_genre,
            target_tag=target_tag,
            target_mood_tag=target_mood_tag,
            target_decade=target_decade,
            preferred_genres=preferred_genres_text,
            preferred_tags=preferred_tags_text,
            parser_confidence=float(confidence),
        )
