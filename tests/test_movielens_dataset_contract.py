from pathlib import Path

import pandas as pd

from media_search_reliability.data_ingestion.load_movielens import load_movielens
from media_search_reliability.evaluation.dataset_smoke import (
    _write_csv_fixture,
    _write_dat_fixture,
)


def _frames():
    movies = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "title": ["Alpha (2000)", "Beta (2001)", "Gamma (2002)"],
            "genres": ["Comedy|Drama", "Action", "Sci-Fi"],
        }
    )
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [1, 2, 2, 3],
            "rating": [4.0, 3.0, 5.0, 4.0],
        }
    )
    tags = pd.DataFrame(
        {
            "user_id": [1, 2],
            "movie_id": [1, 3],
            "tag": ["funny", "space"],
        }
    )
    return movies, ratings, tags


def test_load_movielens_1m_dat_without_tags(tmp_path: Path):
    movies, ratings, _ = _frames()
    _write_dat_fixture(tmp_path, movies, ratings)
    loaded_movies, loaded_ratings, loaded_tags = load_movielens(tmp_path)
    assert list(loaded_movies.columns) == ["movie_id", "title", "genres"]
    assert len(loaded_movies) == 3
    assert len(loaded_ratings) == 4
    assert loaded_tags.empty
    assert list(loaded_tags.columns) == ["user_id", "movie_id", "tag", "timestamp"]


def test_load_modern_movielens_csv_with_tags(tmp_path: Path):
    movies, ratings, tags = _frames()
    _write_csv_fixture(tmp_path, movies, ratings, tags)
    loaded_movies, loaded_ratings, loaded_tags = load_movielens(tmp_path)
    assert len(loaded_movies) == 3
    assert len(loaded_ratings) == 4
    assert len(loaded_tags) == 2
    assert set(loaded_tags["tag"]) == {"funny", "space"}


def test_resolve_nested_movielens_directory(tmp_path: Path):
    from media_search_reliability.data_ingestion.load_movielens import resolve_movielens_directory

    movies, ratings, tags = _frames()
    nested = tmp_path / "download" / "ml-20m"
    _write_csv_fixture(nested, movies, ratings, tags)
    resolved = resolve_movielens_directory(tmp_path)
    assert resolved == nested.resolve()


def test_resolve_missing_movielens_directory_raises(tmp_path: Path):
    import pytest
    from media_search_reliability.data_ingestion.load_movielens import resolve_movielens_directory

    with pytest.raises(FileNotFoundError):
        resolve_movielens_directory(tmp_path)
