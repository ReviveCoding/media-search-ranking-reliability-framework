from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def norm_title(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\(\d{4}\)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def year_from_title(value: object):
    m = re.search(r"\((\d{4})\)", str(value))
    return int(m.group(1)) if m else None


def find_file(root: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p).lower()))
        if matches:
            return matches[0]
    return None


def read_movies(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dat":
        return pd.read_csv(path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin-1")
    return pd.read_csv(path).rename(columns={"movie_id": "movieId"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--catalog", default="data/processed/media_catalog.csv")
    ap.add_argument("--output", default="data/processed/tag_genome_enrichment.csv")
    ap.add_argument("--report", default="reports/24_tag_genome_mapping_report.json")
    ap.add_argument("--top-tags", type=int, default=24)
    ap.add_argument("--min-relevance", type=float, default=0.15)
    args = ap.parse_args()

    root = Path(args.genome_dir)
    catalog = pd.read_csv(args.catalog)
    scores_path = find_file(root, ["genome-scores.csv", "genome_scores.csv", "tag_relevance.dat"])
    tags_path = find_file(root, ["genome-tags.csv", "genome_tags.csv", "tags.json", "tags.csv"])
    movies_path = find_file(root, ["movies.csv", "movies.dat"])
    if not scores_path or not tags_path:
        raise FileNotFoundError("Could not locate genome scores/tags files under the supplied directory")

    scores = pd.read_csv(scores_path)
    scores = scores.rename(columns={"movieId": "genome_movie_id", "movie_id": "genome_movie_id", "tagId": "tag_id", "tagId": "tag_id"})
    if "tag_id" not in scores.columns and "tagId" in scores.columns:
        scores = scores.rename(columns={"tagId": "tag_id"})
    tags = pd.read_csv(tags_path)
    tags = tags.rename(columns={"tagId": "tag_id"})
    if "tag" not in tags.columns:
        raise ValueError("Genome tag file must contain tag text")

    mapping = pd.DataFrame()
    if movies_path:
        gm = read_movies(movies_path)
        gm["norm_title"] = gm["title"].map(norm_title)
        gm["year"] = gm["title"].map(year_from_title)
        cat = catalog[["movie_id", "clean_title", "year"]].copy()
        cat["norm_title"] = cat["clean_title"].map(norm_title)
        mapping = cat.merge(gm[["movieId", "norm_title", "year"]], on=["norm_title", "year"], how="left")
        mapping = mapping.rename(columns={"movieId": "genome_movie_id"})
        mapping["mapping_method"] = mapping["genome_movie_id"].notna().map({True: "title_year", False: "unmatched"})
    else:
        overlap = set(catalog["movie_id"].astype(int)) & set(scores["genome_movie_id"].astype(int).unique())
        mapping = catalog[["movie_id"]].copy()
        mapping["genome_movie_id"] = mapping["movie_id"].where(mapping["movie_id"].isin(overlap))
        mapping["mapping_method"] = mapping["genome_movie_id"].notna().map({True: "direct_id", False: "unmatched"})

    usable = scores[scores["relevance"] >= args.min_relevance].merge(tags[["tag_id", "tag"]], on="tag_id", how="left")
    usable = usable.sort_values(["genome_movie_id", "relevance", "tag"], ascending=[True, False, True])
    top = usable.groupby("genome_movie_id", sort=True).head(args.top_tags)
    agg = top.groupby("genome_movie_id").agg(
        tag_genome_text=("tag", lambda x: "|".join(str(v) for v in x if pd.notna(v))),
        tag_genome_max_relevance=("relevance", "max"),
        tag_genome_tag_count=("tag", "count"),
    ).reset_index()
    out = mapping.merge(agg, on="genome_movie_id", how="left")
    out = out[["movie_id", "tag_genome_text", "tag_genome_max_relevance", "tag_genome_tag_count", "mapping_method"]]
    out = out.rename(columns={"mapping_method": "tag_genome_mapping_method"})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    report = {
        "catalog_movies": int(len(catalog)),
        "mapped_movies": int(out["tag_genome_text"].fillna("").str.len().gt(0).sum()),
        "coverage": float(out["tag_genome_text"].fillna("").str.len().gt(0).mean()),
        "scores_file": str(scores_path),
        "tags_file": str(tags_path),
        "movies_file": str(movies_path) if movies_path else None,
        "output": str(Path(args.output).resolve()),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
