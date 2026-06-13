from __future__ import annotations

from pathlib import Path

from media_search_reliability.data_ingestion.load_movielens import load_movielens


def test_dat_loader_filters_during_chunked_read(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "movies.dat").write_text(
        "1::One (2000)::Drama\n2::Two (2001)::Comedy\n3::Three (2002)::Action\n",
        encoding="latin-1",
    )
    (tmp_path / "ratings.dat").write_text(
        "10::1::4.0::1\n10::2::3.5::2\n20::1::5.0::3\n20::3::2.0::4\n30::1::4.5::5\n",
        encoding="latin-1",
    )
    (tmp_path / "tags.dat").write_text(
        "10::1::thoughtful::1\n20::3::fast::2\n30::1::classic::3\n",
        encoding="latin-1",
    )
    monkeypatch.setenv("MOVIELENS_READ_CHUNKSIZE", "10000")

    movies, ratings, tags = load_movielens(tmp_path, max_movies=2, max_users=2)

    assert movies["movie_id"].tolist() == [1, 2]
    assert set(ratings["user_id"].tolist()) == {10, 20}
    assert set(ratings["movie_id"].tolist()) <= {1, 2}
    assert set(tags["user_id"].tolist()) <= {10, 20}
    assert set(tags["movie_id"].tolist()) <= {1, 2}
