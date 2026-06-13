import pandas as pd

from media_search_reliability.query_understanding import QueryUnderstanding


def _catalog():
    return pd.DataFrame({
        "movie_id": [1, 2],
        "clean_title": ["Hidden Planet", "Rainy Case"],
        "genres": ["Sci-Fi|Comedy", "Crime|Thriller"],
        "tag_text": ["robots|space|funny", "detective|rainy city|dark"],
    })


def test_query_parser_restores_serving_features():
    parser = QueryUnderstanding(_catalog())
    parsed = parser.parse("funny sci-fi movie with robots")
    assert parsed.query_type == "genre_tag"
    assert parsed.target_genre == "Sci-Fi"
    assert parsed.target_tag == "robot"
    assert parsed.target_mood_tag == "funny"


def test_similar_to_parser_resolves_anchor_and_metadata():
    parser = QueryUnderstanding(_catalog())
    parsed = parser.parse("movie like Hidden Planet")
    assert parsed.query_type == "similar_to"
    assert parsed.anchor_movie_id == 1
    assert parsed.target_genre == "Sci-Fi"
    assert parsed.target_tag in {"robot", "space", "funny"}
