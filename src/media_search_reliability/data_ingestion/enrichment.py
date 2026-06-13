from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd

from media_search_reliability.text_normalization import canonical_tag


def _read_optional(path_value: str | None) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Enrichment file does not exist: {path}")
    return pd.read_csv(path)


def _canonical_pipe(value: object) -> str:
    return "|".join(
        dict.fromkeys(
            canonical_tag(item)
            for item in str(value or "").split("|")
            if canonical_tag(item)
        )
    )


def apply_optional_enrichment(catalog: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach optional metadata without silently changing the benchmark definition.

    mode="feature_only" keeps retrieval text and base tags unchanged. This is the
    default for fair ablations because queries, judgments, and retrieval documents
    remain fixed while only numeric ranking features change.
    """
    config = config or {}
    mode = str(config.get("mode", "feature_only"))
    top_text_tags = int(config.get("top_text_tags", 5))
    tag_path = os.environ.get("TAG_GENOME_ENRICHMENT_PATH") or config.get("tag_genome_enrichment_path")
    imdb_path = os.environ.get("IMDB_ENRICHMENT_PATH") or config.get("imdb_enrichment_path")
    out = catalog.copy()
    out["base_tag_text"] = out.get("tag_text", pd.Series("", index=out.index)).fillna("").astype(str)
    base_content_text = out.get("content_text", pd.Series("", index=out.index)).fillna("").astype(str)
    diagnostics = {
        "mode": mode,
        "tag_genome_applied": False,
        "imdb_applied": False,
        "retrieval_text_changed": False,
    }

    tag_df = _read_optional(tag_path)
    if len(tag_df):
        required = {"movie_id", "tag_genome_text"}
        if not required.issubset(tag_df.columns):
            raise ValueError(f"Tag Genome enrichment must contain {sorted(required)}")
        keep = [
            c for c in (
                "movie_id", "tag_genome_text", "tag_genome_max_relevance",
                "tag_genome_tag_count", "tag_genome_mapping_method",
            ) if c in tag_df.columns
        ]
        out = out.merge(tag_df[keep].drop_duplicates("movie_id"), on="movie_id", how="left")
        out["tag_genome_text"] = out["tag_genome_text"].fillna("").map(_canonical_pipe)
        out["tag_genome_coverage"] = out["tag_genome_text"].str.len().gt(0).astype(int)
        diagnostics.update({
            "tag_genome_applied": True,
            "tag_genome_coverage": float(out["tag_genome_coverage"].mean()),
            "tag_genome_path": str(Path(tag_path).resolve()),
        })
    else:
        out["tag_genome_text"] = ""
        out["tag_genome_coverage"] = 0

    imdb_df = _read_optional(imdb_path)
    if len(imdb_df):
        if "movie_id" not in imdb_df.columns:
            raise ValueError("IMDb enrichment must contain movie_id")
        keep = [
            c for c in (
                "movie_id", "imdb_tconst", "imdb_rating", "imdb_num_votes",
                "imdb_runtime_minutes", "imdb_genres", "imdb_mapping_method",
            ) if c in imdb_df.columns
        ]
        out = out.merge(imdb_df[keep].drop_duplicates("movie_id"), on="movie_id", how="left")
        out["imdb_rating"] = pd.to_numeric(out.get("imdb_rating"), errors="coerce")
        out["imdb_num_votes"] = pd.to_numeric(out.get("imdb_num_votes"), errors="coerce").fillna(0)
        out["imdb_runtime_minutes"] = pd.to_numeric(out.get("imdb_runtime_minutes"), errors="coerce")
        out["imdb_votes_log"] = out["imdb_num_votes"].map(lambda x: math.log1p(float(x)))
        out["imdb_genres"] = out.get("imdb_genres", pd.Series("", index=out.index)).fillna("").astype(str)
        out["imdb_coverage"] = out.get("imdb_tconst", pd.Series("", index=out.index)).fillna("").astype(str).str.len().gt(0).astype(int)
        diagnostics.update({
            "imdb_applied": True,
            "imdb_coverage": float(out["imdb_coverage"].mean()),
            "imdb_path": str(Path(imdb_path).resolve()),
        })
    else:
        out["imdb_rating"] = pd.to_numeric(out.get("rating_mean", 3.0), errors="coerce")
        out["imdb_num_votes"] = 0.0
        out["imdb_runtime_minutes"] = float("nan")
        out["imdb_votes_log"] = 0.0
        out["imdb_genres"] = ""
        out["imdb_coverage"] = 0

    if mode == "legacy_append_all":
        out["tag_text"] = out.apply(
            lambda row: "|".join(dict.fromkeys([
                *[canonical_tag(x) for x in str(row.get("base_tag_text", "")).split("|") if canonical_tag(x)],
                *[canonical_tag(x) for x in str(row.get("tag_genome_text", "")).split("|") if canonical_tag(x)],
            ])),
            axis=1,
        )
        out["content_text"] = (
            out["clean_title"].astype(str) + " "
            + out["genres"].astype(str).str.replace("|", " ", regex=False) + " "
            + out["tag_text"].astype(str).str.replace("|", " ", regex=False) + " "
            + out["imdb_genres"].astype(str).str.replace(",", " ", regex=False)
        ).str.strip()
        diagnostics["retrieval_text_changed"] = True
    elif mode == "append_top_tags":
        top_tags = out["tag_genome_text"].map(
            lambda x: " ".join(str(x).split("|")[: max(0, top_text_tags)])
        )
        out["content_text"] = (base_content_text + " " + top_tags).str.strip()
        out["tag_text"] = out["base_tag_text"]
        diagnostics["retrieval_text_changed"] = bool(top_text_tags > 0 and diagnostics["tag_genome_applied"])
    elif mode == "feature_only":
        out["tag_text"] = out["base_tag_text"]
        out["content_text"] = base_content_text
    else:
        raise ValueError(f"Unsupported enrichment mode: {mode}")

    return out, diagnostics
