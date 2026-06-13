from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


_REQUIRED_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("movies.dat", "ratings.dat"),
    ("movies.csv", "ratings.csv"),
)

_DAT_DTYPES: dict[str, str] = {
    "user_id": "int32",
    "movie_id": "int32",
    "rating": "float32",
    "timestamp": "int64",
    "title": "string",
    "genres": "string",
    "tag": "string",
}
_CSV_DTYPES: dict[str, str] = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
    "title": "string",
    "genres": "string",
    "tag": "string",
}


def _dtype_subset(names: Iterable[str], *, csv_names: bool = False) -> dict[str, str]:
    source = _CSV_DTYPES if csv_names else _DAT_DTYPES
    return {name: source[name] for name in names if name in source}


def _read_dat(path: Path, names: list[str], *, chunksize: int | None = None):
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=names,
        encoding="latin-1",
        dtype=_dtype_subset(names),
        chunksize=chunksize,
    )


def _has_supported_layout(path: Path) -> bool:
    return any((path / movies_name).is_file() and (path / ratings_name).is_file() for movies_name, ratings_name in _REQUIRED_LAYOUTS)


def resolve_movielens_directory(raw_dir: str | Path, max_depth: int = 3) -> Path:
    """Resolve a MovieLens directory, including common nested archive layouts."""
    base = Path(raw_dir).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"MovieLens path does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"MovieLens path is not a directory: {base}")
    if _has_supported_layout(base):
        return base.resolve()

    candidates: list[Path] = []
    base_depth = len(base.parts)
    for candidate in sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: str(p).lower()):
        depth = len(candidate.parts) - base_depth
        if depth > max_depth:
            continue
        if _has_supported_layout(candidate):
            candidates.append(candidate)

    if not candidates:
        expected = " or ".join(f"{m}+{r}" for m, r in _REQUIRED_LAYOUTS)
        raise FileNotFoundError(
            f"No supported MovieLens files found under {base} within depth {max_depth}. "
            f"Expected {expected}."
        )

    candidates.sort(key=lambda p: (len(p.parts) - base_depth, str(p).lower()))
    return candidates[0].resolve()


def _select_users_in_order(values: pd.Series, selected: set[int], limit: int | None) -> None:
    if limit is None or len(selected) >= limit:
        return
    for value in values.drop_duplicates().tolist():
        selected.add(int(value))
        if len(selected) >= limit:
            break


def _read_ratings_chunked(
    path: Path,
    *,
    dat_format: bool,
    allowed_movies: set[int] | None,
    max_users: int | None,
    chunksize: int,
) -> tuple[pd.DataFrame, set[int] | None]:
    selected_users: set[int] | None = set() if max_users else None
    kept: list[pd.DataFrame] = []

    if dat_format:
        iterator = _read_dat(
            path,
            ["user_id", "movie_id", "rating", "timestamp"],
            chunksize=chunksize,
        )
    else:
        iterator = pd.read_csv(
            path,
            dtype=_dtype_subset(["userId", "movieId", "rating", "timestamp"], csv_names=True),
            chunksize=chunksize,
        )

    for chunk in iterator:
        if not dat_format:
            chunk = chunk.rename(columns={"userId": "user_id", "movieId": "movie_id"})
        if allowed_movies is not None:
            chunk = chunk[chunk["movie_id"].isin(allowed_movies)]
        if selected_users is not None:
            _select_users_in_order(chunk["user_id"], selected_users, max_users)
            chunk = chunk[chunk["user_id"].isin(selected_users)]
        if not chunk.empty:
            kept.append(chunk)

    if not kept:
        empty = pd.DataFrame(columns=["user_id", "movie_id", "rating", "timestamp"])
        return empty, selected_users
    return pd.concat(kept, ignore_index=True), selected_users


def _read_tags(
    path: Path,
    *,
    dat_format: bool,
    allowed_movies: set[int] | None,
    allowed_users: set[int] | None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "movie_id", "tag", "timestamp"])
    if dat_format:
        tags = _read_dat(path, ["user_id", "movie_id", "tag", "timestamp"])
    else:
        tags = pd.read_csv(
            path,
            dtype=_dtype_subset(["userId", "movieId", "tag", "timestamp"], csv_names=True),
        ).rename(columns={"userId": "user_id", "movieId": "movie_id"})
    if allowed_movies is not None:
        tags = tags[tags["movie_id"].isin(allowed_movies)]
    if allowed_users is not None:
        tags = tags[tags["user_id"].isin(allowed_users)]
    return tags.reset_index(drop=True)


def load_movielens(raw_dir: str | Path, max_movies: int | None = None, max_users: int | None = None):
    raw_dir = resolve_movielens_directory(raw_dir)
    chunksize = max(10_000, int(os.environ.get("MOVIELENS_READ_CHUNKSIZE", "500000")))
    dat_format = (raw_dir / "movies.dat").exists()

    if dat_format:
        movies = _read_dat(raw_dir / "movies.dat", ["movie_id", "title", "genres"])
        ratings_path = raw_dir / "ratings.dat"
        tags_path = raw_dir / "tags.dat"
    elif (raw_dir / "movies.csv").exists():
        movies = pd.read_csv(
            raw_dir / "movies.csv",
            dtype=_dtype_subset(["movieId", "title", "genres"], csv_names=True),
        ).rename(columns={"movieId": "movie_id"})
        ratings_path = raw_dir / "ratings.csv"
        tags_path = raw_dir / "tags.csv"
    else:
        raise FileNotFoundError(f"No MovieLens files found in {raw_dir}")

    if max_movies:
        movies = movies.head(max_movies).copy()
    movies = movies.reset_index(drop=True)
    allowed_movies = set(movies["movie_id"].astype(int).tolist()) if max_movies else None

    ratings, selected_users = _read_ratings_chunked(
        ratings_path,
        dat_format=dat_format,
        allowed_movies=allowed_movies,
        max_users=max_users,
        chunksize=chunksize,
    )
    tags = _read_tags(
        tags_path,
        dat_format=dat_format,
        allowed_movies=allowed_movies,
        allowed_users=selected_users,
    )

    for df in (movies, ratings, tags):
        for column in df.columns:
            if column.endswith("_id"):
                df[column] = pd.to_numeric(df[column], errors="raise", downcast="integer")
    if "rating" in ratings:
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise", downcast="float")
    return movies, ratings, tags

