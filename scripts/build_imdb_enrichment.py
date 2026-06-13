from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def norm_title(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p).lower()))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imdb-dir", required=True)
    ap.add_argument("--catalog", default="data/processed/media_catalog.csv")
    ap.add_argument("--output", default="data/processed/imdb_enrichment.csv")
    ap.add_argument("--report", default="reports/25_imdb_mapping_report.json")
    args = ap.parse_args()

    root = Path(args.imdb_dir)
    catalog = pd.read_csv(args.catalog)
    basics_path = find(root, "title.basics.tsv.gz")
    ratings_path = find(root, "title.ratings.tsv.gz")
    basics = pd.read_csv(basics_path, sep="\t", na_values="\\N", low_memory=False)
    basics = basics[basics["titleType"].isin(["movie", "tvMovie", "video"])]
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
    basics["norm_title"] = basics["primaryTitle"].map(norm_title)
    basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")
    ratings = pd.read_csv(ratings_path, sep="\t")

    cat = catalog[["movie_id", "clean_title", "year"]].copy()
    cat["norm_title"] = cat["clean_title"].map(norm_title)
    merged = cat.merge(
        basics[["tconst", "norm_title", "startYear", "runtimeMinutes", "genres"]],
        left_on=["norm_title", "year"], right_on=["norm_title", "startYear"], how="left",
    )
    # Deterministically keep the title with the most IMDb votes when duplicates exist.
    merged = merged.merge(ratings, on="tconst", how="left")
    merged["numVotes"] = pd.to_numeric(merged["numVotes"], errors="coerce").fillna(0)
    merged = merged.sort_values(["movie_id", "numVotes", "tconst"], ascending=[True, False, True]).drop_duplicates("movie_id")
    out = merged[["movie_id", "tconst", "averageRating", "numVotes", "runtimeMinutes", "genres"]].rename(columns={
        "tconst": "imdb_tconst", "averageRating": "imdb_rating", "numVotes": "imdb_num_votes",
        "runtimeMinutes": "imdb_runtime_minutes", "genres": "imdb_genres",
    })
    out["imdb_mapping_method"] = out["imdb_tconst"].notna().map({True: "title_year", False: "unmatched"})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    report = {
        "catalog_movies": int(len(catalog)),
        "mapped_movies": int(out["imdb_tconst"].notna().sum()),
        "coverage": float(out["imdb_tconst"].notna().mean()),
        "basics_file": str(basics_path),
        "ratings_file": str(ratings_path),
        "output": str(Path(args.output).resolve()),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
