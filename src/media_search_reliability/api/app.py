from __future__ import annotations

from pathlib import Path
import json
import os
import time
import threading
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from media_search_reliability.features.retrieval_features import build_candidate_features
from media_search_reliability.query_understanding import QueryUnderstanding
from media_search_reliability.ranking.train_lambdarank import predict_ranker
from media_search_reliability.retrieval.hybrid_retriever import HybridRetriever
from media_search_reliability.retrieval.specialized_retriever import SpecializedCandidateGenerator

app = FastAPI(title="Media Search Reliability Framework")

ARTIFACT_DIR = Path(os.environ.get("MEDIA_SEARCH_ARTIFACT_DIR", "artifacts"))
_bundle = None
_catalog = None
_ranker_bundle = None
_hybrid = None
_query_understanding = None
_specialized = None
_load_lock = threading.Lock()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    anchor_movie_id: int | None = Field(default=None, description="Optional item to exclude and use as context for similar-to searches.")
    preferred_genres: list[str] = Field(default_factory=list, max_length=20)
    preferred_tags: list[str] = Field(default_factory=list, max_length=20)


class BatchSearchItem(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    anchor_movie_id: int | None = None
    preferred_genres: list[str] = Field(default_factory=list, max_length=20)
    preferred_tags: list[str] = Field(default_factory=list, max_length=20)


class BatchSearchRequest(BaseModel):
    queries: list[str] | None = Field(default=None, max_length=50)
    items: list[BatchSearchItem] | None = Field(default=None, max_length=50)
    top_k: int = Field(default=10, ge=1, le=50)


def _artifact_readiness() -> dict:
    required = [ARTIFACT_DIR / "retrieval_bundle.joblib", ARTIFACT_DIR / "ranker_bundle.joblib"]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "ready": not missing,
        "missing_artifacts": missing,
        "artifact_dir": str(ARTIFACT_DIR),
        "model_loaded": _bundle is not None,
    }


def _load():
    global _bundle, _catalog, _ranker_bundle, _hybrid, _query_understanding, _specialized
    if _bundle is not None:
        return (_bundle, _ranker_bundle, _catalog, _hybrid, _query_understanding, _specialized), 0.0
    with _load_lock:
        if _bundle is not None:
            return (_bundle, _ranker_bundle, _catalog, _hybrid, _query_understanding, _specialized), 0.0
        load_start = time.perf_counter()
        bundle_path = ARTIFACT_DIR / "retrieval_bundle.joblib"
        ranker_path = ARTIFACT_DIR / "ranker_bundle.joblib"
        if not bundle_path.exists() or not ranker_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts in {ARTIFACT_DIR}. Run media-search-run first "
                "or set MEDIA_SEARCH_ARTIFACT_DIR to the artifact directory."
            )
        _bundle = joblib.load(bundle_path)
        _ranker_bundle = joblib.load(ranker_path)
        _catalog = _bundle.get("catalog")
        if _catalog is None:
            catalog_path = Path(_bundle.get("catalog_path", "data/processed/media_catalog.csv"))
            if not catalog_path.is_absolute():
                catalog_path = ARTIFACT_DIR.parent / catalog_path
            _catalog = pd.read_csv(catalog_path)
        _hybrid = HybridRetriever(_bundle["bm25"], _bundle["dense"], vector_index=_bundle.get("vector_index"))
        _query_understanding = QueryUnderstanding(_catalog)
        _specialized = SpecializedCandidateGenerator(
            _catalog, _bundle["dense"], _bundle.get("vector_index"), user_context=_bundle.get("user_context")
        )
        load_ms = (time.perf_counter() - load_start) * 1000
        return (_bundle, _ranker_bundle, _catalog, _hybrid, _query_understanding, _specialized), load_ms


def _matched_signals(row) -> list[str]:
    signals = []
    for name, threshold in [("bm25", 0.2), ("dense", 0.2), ("hybrid", 0.2)]:
        col = f"{name}_score"
        if float(row.get(col, 0.0)) >= threshold:
            signals.append(name)
    if int(row.get("genre_overlap", 0)) == 1:
        signals.append("genre")
    if int(row.get("tag_overlap", 0)) == 1:
        signals.append("tag")
    if int(row.get("mood_overlap", 0)) == 1:
        signals.append("mood")
    if int(row.get("decade_match", 0)) == 1:
        signals.append("decade")
    if float(row.get("user_genre_affinity", 0.0)) > 0:
        signals.append("user_genre_affinity")
    if float(row.get("user_tag_affinity", 0.0)) > 0:
        signals.append("user_tag_affinity")
    if float(row.get("anchor_dense_score", 0.0)) > 0:
        signals.append("anchor_dense")
    if float(row.get("anchor_metadata_score", 0.0)) > 0:
        signals.append("anchor_metadata")
    if float(row.get("personalized_dense_score", 0.0)) > 0:
        signals.append("personalized_dense")
    if float(row.get("tag_genome_coverage", 0.0)) > 0:
        signals.append("tag_genome")
    if float(row.get("imdb_coverage", 0.0)) > 0:
        signals.append("imdb")
    return signals or ["ranker"]


def _search_impl(
    query: str,
    top_k: int,
    anchor_movie_id: int | None = None,
    preferred_genres: list[str] | None = None,
    preferred_tags: list[str] | None = None,
):
    total_start = time.perf_counter()
    loaded, model_load_ms = _load()
    _, ranker_bundle, catalog, hybrid, parser, specialized = loaded
    scoring_start = time.perf_counter()
    parsed = parser.parse(
        query,
        anchor_movie_id=anchor_movie_id,
        preferred_genres=preferred_genres,
        preferred_tags=preferred_tags,
    )

    hybrid_rows = hybrid.search(query, top_k=max(top_k * 5, 50))
    special_query = type("QueryRow", (), {
        "query_type": parsed.query_type,
        "anchor_movie_id": parsed.anchor_movie_id,
        "user_id": 0,
        "target_genre": parsed.target_genre,
        "target_tag": parsed.target_tag,
        "preferred_genres": parsed.preferred_genres,
        "preferred_tags": parsed.preferred_tags,
    })()
    special = specialized.retrieve(special_query, top_k=max(top_k * 5, 50))
    hybrid_map = {int(movie_id): (float(hy), float(bm), float(de)) for movie_id, hy, bm, de in hybrid_rows}
    specialized_ids = []
    for score_map in special.values():
        specialized_ids.extend(int(mid) for mid in score_map)
    movie_ids = list(dict.fromkeys(list(hybrid_map) + specialized_ids))
    rows = []
    for movie_id in movie_ids:
        if parsed.anchor_movie_id >= 0 and int(movie_id) == int(parsed.anchor_movie_id):
            continue
        hy, bm, de = hybrid_map.get(int(movie_id), (0.0, 0.0, 0.0))
        a_dense = float(special.get("anchor_dense_score", {}).get(int(movie_id), 0.0))
        a_meta = float(special.get("anchor_metadata_score", {}).get(int(movie_id), 0.0))
        p_dense = float(special.get("personalized_dense_score", {}).get(int(movie_id), 0.0))
        rows.append({
            "query_id": 0, "movie_id": movie_id, "hybrid_score": hy, "bm25_score": bm, "dense_score": de,
            "anchor_dense_score": a_dense, "anchor_metadata_score": a_meta,
            "personalized_dense_score": p_dense, "specialized_score": max(a_dense, a_meta, p_dense),
        })

    if not rows:
        return {
            "query": query,
            "parsed_intent": parsed.to_dict(),
            "model_variant": "hybrid_lambdarank_calibrated",
            "results": [],
            "model_load_ms": round(model_load_ms, 3),
            "latency_ms": round((time.perf_counter() - scoring_start) * 1000, 3),
            "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 3),
            "review_flags": ["NO_CANDIDATES"],
        }

    candidates = pd.DataFrame(rows)
    query_df = pd.DataFrame([{
        "query_id": 0,
        "query": query,
        "query_type": parsed.query_type,
        "anchor_movie_id": parsed.anchor_movie_id,
        "user_id": 0,
        "target_genre": parsed.target_genre,
        "target_tag": parsed.target_tag,
        "target_mood_tag": parsed.target_mood_tag,
        "target_decade": parsed.target_decade,
    }])
    labels = pd.DataFrame(columns=["query_id", "movie_id", "label", "slice_long_tail", "slice_cold_start"])
    user = pd.DataFrame([{
        "user_id": 0,
        "preferred_genres": parsed.preferred_genres,
        "preferred_tags": parsed.preferred_tags,
    }])
    feats = build_candidate_features(candidates, query_df, catalog, labels, user)
    feats["score"] = predict_ranker(ranker_bundle["ranker"], feats)
    calibrator = ranker_bundle.get("calibrator")
    feats["calibrated_score"] = calibrator.predict_proba(feats["score"]) if calibrator is not None else feats["score"]

    out = feats.sort_values(
        ["score", "movie_id"], ascending=[False, True], kind="mergesort"
    ).head(top_k).copy()
    results = []
    for _, row in out.iterrows():
        calibrated = float(row.get("calibrated_score", 0.0))
        flags = []
        if calibrated < 0.25:
            flags.append("LOW_CONFIDENCE")
        if int(row.get("long_tail_flag", 0)) == 1:
            flags.append("LONG_TAIL_ITEM")
        results.append({
            "movie_id": int(row["movie_id"]),
            "title": str(row["clean_title"]),
            "genres": str(row["genres"]),
            "score": float(row["score"]),
            "calibrated_score": calibrated,
            "matched_signals": _matched_signals(row),
            "review_flags": flags,
        })

    top_review_flags = sorted({flag for result in results for flag in result["review_flags"]})
    if parsed.parser_confidence < 0.2:
        top_review_flags.append("LOW_QUERY_INTENT_CONFIDENCE")
    return {
        "query": query,
        "parsed_intent": parsed.to_dict(),
        "model_variant": "hybrid_lambdarank_calibrated",
        "results": results,
        "model_load_ms": round(model_load_ms, 3),
        "latency_ms": round((time.perf_counter() - scoring_start) * 1000, 3),
        "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 3),
        "review_flags": sorted(set(top_review_flags)),
    }


@app.get("/health")
def health():
    readiness = _artifact_readiness()
    return {"status": "ok", **readiness}


@app.get("/ready")
def ready():
    readiness = _artifact_readiness()
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail=readiness)
    try:
        _, load_ms = _load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    refreshed = _artifact_readiness()
    return {"status": "ready", **refreshed, "load_latency_ms": round(load_ms, 3)}


@app.post("/search")
def search(req: SearchRequest):
    try:
        return _search_impl(req.query, req.top_k, req.anchor_movie_id, req.preferred_genres, req.preferred_tags)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch_search")
def batch_search(req: BatchSearchRequest):
    if not req.queries and not req.items:
        raise HTTPException(status_code=422, detail="Provide either queries or items.")
    requests = list(req.items or [])
    requests.extend(BatchSearchItem(query=query) for query in (req.queries or []))
    try:
        responses = [
            _search_impl(item.query, req.top_k, item.anchor_movie_id, item.preferred_genres, item.preferred_tags)
            for item in requests
        ]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model_variant": "hybrid_lambdarank_calibrated", "responses": responses}


@app.get("/launch_gate")
def launch_gate():
    path = ARTIFACT_DIR / "launch_gate.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Run the pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/launch_gate")
def launch_gate_post():
    return launch_gate()


@app.post("/evaluate")
def evaluate():
    summary_path = ARTIFACT_DIR / "eval_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=503, detail="Run the pipeline first.")
    return json.loads(summary_path.read_text(encoding="utf-8"))
