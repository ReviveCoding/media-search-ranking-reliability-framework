"""Contract tests for order-preserving canonical value joining."""

from media_search_reliability.text_normalization import canonical_join


def test_canonical_join_keeps_first_genre_occurrence() -> None:
    genres = ["Sci-Fi", "Comedy", "science fiction", "Sci Fi", "Comedy"]
    assert canonical_join(genres, kind="genre") == "sci-fi|comedy"


def test_canonical_join_deduplicates_tag_aliases_without_reordering() -> None:
    tags = ["space", "Robots", "robotic", "space", "", "funny"]
    assert canonical_join(tags, kind="tag") == "space|robot|funny"


def test_canonical_join_accepts_empty_input() -> None:
    assert canonical_join([], kind="genre") == ""
    assert canonical_join([], kind="tag") == ""
