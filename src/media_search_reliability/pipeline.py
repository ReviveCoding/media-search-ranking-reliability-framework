from __future__ import annotations

from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd
import joblib

from media_search_reliability.utils import ensure_dir, load_yaml, seed_everything, save_table, write_json, normalize_scores
from media_search_reliability.data_ingestion.synthetic_data import generate_synthetic_movielens
from media_search_reliability.data_ingestion.load_movielens import load_movielens, resolve_movielens_directory
from media_search_reliability.data_ingestion.build_media_catalog import build_media_catalog
from media_search_reliability.data_ingestion.validate_catalog import validate_catalog, write_validation_report
from media_search_reliability.data_ingestion.enrichment import apply_optional_enrichment
from media_search_reliability.query_labeling.generate_queries import generate_queries_and_labels
from media_search_reliability.retrieval.bm25_retriever import BM25Retriever
from media_search_reliability.retrieval.dense_retriever import DenseRetriever
from media_search_reliability.retrieval.faiss_index import VectorIndex
from media_search_reliability.retrieval.hybrid_retriever import HybridRetriever
from media_search_reliability.retrieval.specialized_retriever import SpecializedCandidateGenerator
from media_search_reliability.features.retrieval_features import build_candidate_features, FEATURE_COLUMNS
from media_search_reliability.ranking.train_lambdarank import train_lambdarank, predict_ranker
from media_search_reliability.ranking.calibrate_scores import ScoreCalibrator
from media_search_reliability.evaluation.ranking_metrics import evaluate_variants, metrics_at_k
from media_search_reliability.evaluation.calibration_metrics import expected_calibration_error, brier_score
from media_search_reliability.evaluation.slice_eval import evaluate_slices
from media_search_reliability.evaluation.latency_eval import measure_search_latency
from media_search_reliability.evaluation.launch_gate import evaluate_launch_gate
from media_search_reliability.evaluation.splitting import split_queries_group_aware
from media_search_reliability.evaluation.frozen_manifest import save_frozen_manifest, load_frozen_manifest
from media_search_reliability.evaluation.candidate_diagnostics import candidate_recall_diagnostics
from media_search_reliability.reporting.make_reports import write_report, _md_table, write_model_card, write_claim_boundary


def _split_queries(queries: pd.DataFrame, test_size: float, val_size: float, seed: int = 42, return_diagnostics: bool = False):
    """Group-aware split that prevents anchor and personalized-user leakage."""
    result = split_queries_group_aware(queries, test_size=test_size, val_size=val_size, seed=seed)
    payload = (result.train_query_ids, result.val_query_ids, result.test_query_ids)
    return (*payload, result.diagnostics) if return_diagnostics else payload


def _build_candidate_table(
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    bm25,
    dense,
    hybrid,
    top_k: int,
    specialized=None,
    inject_relevant_labels: bool = False,
    max_injected_per_query: int = 12,
    candidate_quotas: dict | None = None,
):
    """Build leakage-controlled candidates with optional query-type specialization."""
    rows = []
    candidate_quotas = candidate_quotas or {}
    default_quotas = candidate_quotas.get("default", {}) if isinstance(candidate_quotas, dict) else {}
    for query_row in queries.itertuples():
        query_type = str(getattr(query_row, "query_type", "default"))
        query_quotas = candidate_quotas.get(query_type, {}) if isinstance(candidate_quotas, dict) else {}
        quotas = {**default_quotas, **query_quotas}
        bm25_k = int(quotas.get("bm25", top_k))
        dense_k = int(quotas.get("dense", top_k))
        hybrid_k = int(quotas.get("hybrid", top_k))
        specialized_k = int(quotas.get("specialized", top_k))
        bm = bm25.search(query_row.query, bm25_k) if bm25_k > 0 else []
        de = dense.search(query_row.query, dense_k) if dense_k > 0 else []
        hy = hybrid.search(query_row.query, hybrid_k) if hybrid_k > 0 else []
        special = specialized.retrieve(query_row, specialized_k) if specialized is not None and specialized_k > 0 else {
            "anchor_dense_score": {}, "anchor_metadata_score": {}, "personalized_dense_score": {}
        }
        specialized_ids = []
        for score_map in special.values():
            specialized_ids.extend(score_map.keys())
        movie_ids = list(dict.fromkeys(
            [doc for doc, _ in bm] + [doc for doc, _ in de] + [doc for doc, *_ in hy] + specialized_ids
        ))

        if str(getattr(query_row, "query_type", "")) == "similar_to":
            anchor_id = int(getattr(query_row, "anchor_movie_id", -1))
            movie_ids = [movie_id for movie_id in movie_ids if int(movie_id) != anchor_id]

        if inject_relevant_labels:
            positives = labels[(labels["query_id"] == query_row.query_id) & (labels["label"] >= 1)].copy()
            positives = positives.sort_values(["label", "movie_id"], ascending=[False, True])
            missing = [int(mid) for mid in positives["movie_id"].tolist() if int(mid) not in movie_ids]
            movie_ids.extend(missing[:max_injected_per_query])

        bm_map, de_map = dict(bm), dict(de)
        hy_map = {doc: score for doc, score, _, _ in hy}
        bm_vals = normalize_scores(np.array([bm_map.get(doc, 0.0) for doc in movie_ids]))
        de_vals = normalize_scores(np.array([de_map.get(doc, 0.0) for doc in movie_ids]))
        hy_vals = normalize_scores(np.array([hy_map.get(doc, 0.0) for doc in movie_ids]))
        anchor_dense = special.get("anchor_dense_score", {})
        anchor_metadata = special.get("anchor_metadata_score", {})
        personalized_dense = special.get("personalized_dense_score", {})
        for i, movie_id in enumerate(movie_ids):
            a_dense = float(anchor_dense.get(int(movie_id), 0.0))
            a_meta = float(anchor_metadata.get(int(movie_id), 0.0))
            p_dense = float(personalized_dense.get(int(movie_id), 0.0))
            rows.append({
                "query_id": int(query_row.query_id),
                "movie_id": int(movie_id),
                "bm25_score": float(bm_vals[i]),
                "dense_score": float(de_vals[i]),
                "hybrid_score": float(hy_vals[i]),
                "anchor_dense_score": a_dense,
                "anchor_metadata_score": a_meta,
                "personalized_dense_score": p_dense,
                "specialized_score": max(a_dense, a_meta, p_dense),
                "training_label_injected": int(movie_id not in bm_map and movie_id not in de_map and movie_id not in hy_map and int(movie_id) not in specialized_ids),
            })
    columns = [
        "query_id", "movie_id", "bm25_score", "dense_score", "hybrid_score",
        "anchor_dense_score", "anchor_metadata_score", "personalized_dense_score", "specialized_score",
        "training_label_injected",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).drop_duplicates(["query_id", "movie_id"])


def _search_with_ranker(query, catalog, bm25, dense, hybrid, ranker, top_k=10):
    temp_query = pd.DataFrame([{
        "query_id": -1, "query": query, "query_type": "adhoc", "anchor_movie_id": -1,
        "user_id": 0, "target_genre": "", "target_tag": "", "target_decade": -1,
    }])
    empty_labels = pd.DataFrame(columns=["query_id", "movie_id", "label", "slice_long_tail", "slice_cold_start"])
    specialized = SpecializedCandidateGenerator(catalog, dense, hybrid.vector_index, user_context=pd.DataFrame({"user_id": [0], "preferred_genres": [""], "preferred_tags": [""]}))
    candidates = _build_candidate_table(temp_query, empty_labels, bm25, dense, hybrid, top_k=min(80, len(catalog)), specialized=specialized)
    features = build_candidate_features(
        candidates, temp_query, catalog, empty_labels,
        user_context=pd.DataFrame({"user_id": [0], "preferred_genres": [""], "preferred_tags": [""]}),
    )
    features["ranker_score"] = predict_ranker(ranker, features)
    return features.sort_values(
        ["ranker_score", "movie_id"], ascending=[False, True], kind="mergesort"
    ).head(top_k)[["movie_id", "clean_title", "ranker_score", "genres"]]



def _similar_to_self_return_rate(df: pd.DataFrame, score_col: str = "ranker_score", k: int = 10) -> float:
    subset = df[df.get("query_type", pd.Series(index=df.index, dtype=str)).astype(str) == "similar_to"]
    if subset.empty:
        return 0.0
    returned = []
    for _, group in subset.groupby("query_id"):
        anchor_values = pd.to_numeric(group.get("anchor_movie_id", pd.Series(dtype=float)), errors="coerce").dropna()
        if anchor_values.empty:
            continue
        anchor_id = int(anchor_values.iloc[0])
        top_ids = group.sort_values(
            [score_col, "movie_id"], ascending=[False, True], kind="mergesort"
        ).head(k)["movie_id"].astype(int).tolist()
        returned.append(float(anchor_id in top_ids))
    return float(np.mean(returned)) if returned else 0.0

def _gini(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or np.allclose(arr.sum(), 0):
        return 0.0
    arr = np.sort(np.clip(arr, 0, None))
    n = len(arr)
    return float((2 * np.sum((np.arange(1, n + 1)) * arr) / (n * arr.sum())) - (n + 1) / n)


def run_pipeline(config_path: str | Path, mode: str = "demo") -> dict:
    def log(message):
        print(f"[pipeline] {message}", flush=True)

    cfg = load_yaml(config_path)
    seed = int(cfg["project"].get("random_seed", 42))
    seed_everything(seed)
    root = Path.cwd()
    data_dir = ensure_dir(root / cfg["project"].get("data_dir", "data"))
    processed_dir = ensure_dir(data_dir / "processed")
    synthetic_dir = ensure_dir(data_dir / "synthetic")
    out_dir = ensure_dir(root / cfg["project"].get("output_dir", "artifacts"))
    reports_dir = ensure_dir(root / cfg["project"].get("reports_dir", "reports"))

    log("loading data")
    if mode == "movielens":
        configured_raw_dir = os.environ.get("MOVIELENS_RAW_DIR") or cfg["data"].get("movielens_raw_dir")
        requested_raw_dir = Path(configured_raw_dir).expanduser() if configured_raw_dir else data_dir / "raw" / "movielens"
        raw_dir = resolve_movielens_directory(requested_raw_dir)
        movies, ratings, tags = load_movielens(
            raw_dir,
            max_movies=cfg["data"].get("max_movies"),
            max_users=cfg["data"].get("max_users"),
        )
        data_source = str(raw_dir.resolve())
    else:
        synthetic_cfg = cfg["data"].get("synthetic", {})
        movies, ratings, tags = generate_synthetic_movielens(
            n_movies=int(cfg["data"].get("demo_movies", 1200)),
            n_users=int(cfg["data"].get("demo_users", 800)),
            n_ratings=int(cfg["data"].get("demo_ratings", 40000)),
            seed=seed,
            popularity_alpha=float(synthetic_cfg.get("popularity_alpha", 1.05)),
            preference_strength=float(synthetic_cfg.get("preference_strength", 1.35)),
            rating_noise=float(synthetic_cfg.get("rating_noise", 0.70)),
            tag_observation_prob=float(synthetic_cfg.get("tag_observation_prob", 0.45)),
            exploration_rate=float(synthetic_cfg.get("exploration_rate", 0.12)),
            cold_start_fraction=float(synthetic_cfg.get("cold_start_fraction", 0.08)),
        )
        data_source = "monte-carlo-synthetic-demo"

    log("building catalog")
    base_catalog = build_media_catalog(movies, ratings, tags)

    benchmark_cfg = cfg.get("benchmark", {}) or {}
    manifest_dir_value = os.environ.get("FROZEN_BENCHMARK_DIR") or benchmark_cfg.get("manifest_dir")
    manifest_mode = str(os.environ.get("FROZEN_BENCHMARK_MODE") or benchmark_cfg.get("manifest_mode", "off")).lower()
    manifest_dir = Path(manifest_dir_value).expanduser() if manifest_dir_value else None
    manifest_metadata = None

    log("generating or loading frozen queries and labels")
    label_all_catalog = bool(cfg["queries"].get("label_all_catalog_demo", True)) if mode == "demo" else bool(cfg["queries"].get("label_all_catalog_movielens", False))
    should_reuse = manifest_dir is not None and manifest_mode in {"reuse", "auto"} and (manifest_dir / "manifest.json").exists()
    if should_reuse:
        frozen = load_frozen_manifest(
            manifest_dir,
            catalog=base_catalog,
            require_catalog_fingerprint=bool(benchmark_cfg.get("require_catalog_fingerprint", True)),
        )
        queries = frozen.queries
        labels = frozen.labels
        user_context = frozen.user_context
        train_q, val_q, test_q = frozen.train_query_ids, frozen.val_query_ids, frozen.test_query_ids
        manifest_metadata = frozen.metadata
        split_diagnostics = {
            "strategy": "frozen-benchmark-manifest",
            "split_sizes": {"train": len(train_q), "val": len(val_q), "test": len(test_q)},
            "anchor_leakage_free": True,
            "personalized_user_leakage_free": True,
            "anchor_overlap": {},
            "personalized_user_overlap": {},
            "query_type_counts": {
                split: queries[queries["query_id"].isin(ids)]["query_type"].value_counts().to_dict()
                for split, ids in (("train", train_q), ("val", val_q), ("test", test_q))
            },
        }
    else:
        queries, labels, user_context = generate_queries_and_labels(
            base_catalog,
            ratings,
            num_queries=int(cfg["queries"].get("num_queries", 450)),
            candidates_per_query=int(cfg["queries"].get("candidates_per_query", 80)),
            seed=seed,
            label_all_catalog=label_all_catalog,
            label_noise=float(cfg["queries"].get("label_noise", 0.0)),
        )
        train_q, val_q, test_q, split_diagnostics = _split_queries(
            queries,
            cfg["ranking"].get("test_size", 0.20),
            cfg["ranking"].get("val_size", 0.15),
            seed=seed,
            return_diagnostics=True,
        )
        if manifest_dir is not None and manifest_mode in {"create", "auto"}:
            save_frozen_manifest(
                manifest_dir,
                queries=queries,
                labels=labels,
                user_context=user_context,
                train_query_ids=train_q,
                val_query_ids=val_q,
                test_query_ids=test_q,
                catalog=base_catalog,
                seed=seed,
                data_source=data_source,
            )
            manifest_metadata = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))

    catalog, enrichment_diagnostics = apply_optional_enrichment(base_catalog, cfg.get("enrichment", {}))
    save_table(catalog, processed_dir / "media_catalog.csv")
    save_table(ratings, processed_dir / "interactions.csv")
    save_table(tags, processed_dir / "tags.csv")
    save_table(queries, synthetic_dir / "synthetic_queries.csv")
    save_table(labels, processed_dir / "graded_relevance_labels.csv")
    save_table(user_context, synthetic_dir / "synthetic_user_context.csv")
    stats = validate_catalog(catalog, ratings, tags)
    write_validation_report(stats, reports_dir / "01_data_validation_report.md")

    label_dist = labels["label"].value_counts().sort_index().rename_axis("label").reset_index(name="count")
    query_dist = queries["query_type"].value_counts().rename_axis("query_type").reset_index(name="count")
    positive_by_query = labels.groupby("query_id")["label"].agg(max_label="max", positive_count=lambda s: int((s >= 2).sum())).reset_index()
    write_report(reports_dir / "10_query_label_quality_report.md", "Query and Label Quality Report", {
        "Judgment coverage": "Complete catalog judgments for demo mode." if label_all_catalog else "Pooled proxy judgments for memory-efficient MovieLens mode.",
        "Frozen benchmark": json.dumps(manifest_metadata or {"enabled": False}, indent=2),
        "Query-type coverage": _md_table(query_dist),
        "Graded label distribution": _md_table(label_dist),
        "Positive-label coverage": _md_table(positive_by_query.describe().reset_index()),
        "Interpretation": "Labels remain public/synthetic proxy judgments rather than human search judgments. Frozen ablations reuse identical queries, judgments, and splits.",
    })

    train_queries = queries[queries["query_id"].isin(train_q)]

    log("fitting retrieval models")
    docs = catalog["content_text"].fillna("").astype(str).tolist()
    doc_ids = catalog["movie_id"].astype(int).tolist()
    bm25 = BM25Retriever().fit(docs, doc_ids=doc_ids)
    dense = DenseRetriever(
        backend=cfg["retrieval"].get("dense_backend", "auto"),
        model_name=cfg["retrieval"].get("dense_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=cfg["retrieval"].get("dense_device", "auto"),
        batch_size=int(cfg["retrieval"].get("dense_batch_size", 64)),
        allow_backend_fallback=bool(cfg["retrieval"].get("allow_backend_fallback", True)),
        show_progress_bar=bool(cfg["retrieval"].get("show_progress_bar", False)),
    ).fit(docs, doc_ids=doc_ids)
    vector_index = VectorIndex(use_gpu_if_available=bool(cfg["retrieval"].get("faiss_use_gpu_if_available", True))).fit(dense.embeddings, doc_ids)
    hybrid = HybridRetriever(bm25, dense, alpha=float(cfg["retrieval"].get("hybrid_alpha", 0.55)), vector_index=vector_index)
    specialized = SpecializedCandidateGenerator(catalog, dense, vector_index, user_context=user_context)

    top_k = int(cfg["retrieval"].get("top_k", 80))
    log("building leakage-controlled candidate tables")
    candidate_quotas = cfg["retrieval"].get("candidate_quotas", {})
    natural_candidates = _build_candidate_table(
        queries, labels, bm25, dense, hybrid, top_k=top_k, specialized=specialized,
        inject_relevant_labels=False, candidate_quotas=candidate_quotas,
    )
    train_candidates = _build_candidate_table(
        train_queries, labels, bm25, dense, hybrid, top_k=top_k, specialized=specialized,
        inject_relevant_labels=True, candidate_quotas=candidate_quotas,
    )
    save_table(natural_candidates, processed_dir / "serving_candidates.csv")
    save_table(train_candidates, processed_dir / "training_candidates.csv")
    test_candidate_diagnostics = candidate_recall_diagnostics(
        natural_candidates[natural_candidates["query_id"].isin(test_q)],
        labels[labels["query_id"].isin(test_q)],
        queries[queries["query_id"].isin(test_q)],
        positive_label_min=int(cfg.get("evaluation", {}).get("positive_label_min", 2)),
        k_values=(50, 100),
    )

    log("building features")
    train_df = build_candidate_features(train_candidates, train_queries, catalog, labels, user_context=user_context).sort_values("query_id")
    natural_features = build_candidate_features(natural_candidates, queries, catalog, labels, user_context=user_context)
    val_df = natural_features[natural_features["query_id"].isin(val_q)].sort_values("query_id").copy()
    test_df = natural_features[natural_features["query_id"].isin(test_q)].sort_values("query_id").copy()
    save_table(pd.concat([train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")], ignore_index=True), processed_dir / "candidate_features.csv")

    log("training ranker")
    ranker = train_lambdarank(train_df, val_df, feature_columns=FEATURE_COLUMNS, config=cfg.get("ranking", {}))
    for frame in (train_df, val_df, test_df):
        frame["ranker_score"] = predict_ranker(ranker, frame)

    log("calibrating and evaluating")
    positive_label_min = int(cfg.get("evaluation", {}).get("positive_label_min", 2))
    calibrator = ScoreCalibrator(
        method=cfg.get("calibration", {}).get("method", "auto"),
        positive_label_min=int(cfg.get("calibration", {}).get("positive_label_min", positive_label_min)),
    ).fit(val_df["ranker_score"], val_df["label"])
    test_df["calibrated_score"] = calibrator.predict_proba(test_df["ranker_score"])

    # In synthetic mode the simulator exposes latent clean relevance. Models train
    # on observed/noisy judgments, while final stress metrics use clean latent truth.
    # This prevents label noise from contaminating both training and evaluation.
    evaluation_label_col = "clean_label" if mode == "demo" and "clean_label" in labels.columns else "label"
    eval_test_df = test_df.copy()
    if evaluation_label_col in eval_test_df.columns:
        eval_test_df["label"] = eval_test_df[evaluation_label_col].astype(int)
    test_truth = labels[labels["query_id"].isin(test_q)].copy()
    observed_test_truth = test_truth.copy()
    if evaluation_label_col in test_truth.columns:
        test_truth["label"] = test_truth[evaluation_label_col].astype(int)

    # Scenario diagnostic: evaluate retrieval-only visual-query behavior across
    # all generated visual queries. This avoids a one-query test slice dominating
    # the Monte Carlo sparse-metadata direction check while keeping the learned
    # ranker test metrics strictly held out.
    visual_query_ids = set(queries.loc[queries["query_type"] == "visual_query", "query_id"].astype(int))
    visual_features_all = natural_features[natural_features["query_id"].isin(visual_query_ids)].copy()
    visual_truth_all = labels[labels["query_id"].isin(visual_query_ids)].copy()
    if evaluation_label_col in visual_features_all.columns:
        visual_features_all["label"] = visual_features_all[evaluation_label_col].astype(int)
    if evaluation_label_col in visual_truth_all.columns:
        visual_truth_all["label"] = visual_truth_all[evaluation_label_col].astype(int)
    visual_hybrid_metrics_all = metrics_at_k(
        visual_features_all,
        "hybrid_score",
        k=10,
        positive_label_min=positive_label_min,
        ground_truth=visual_truth_all,
    ) if len(visual_features_all) else {"ndcg_at_10": 0.0}
    diagnostic_metrics = {
        "visual_query_hybrid_ndcg_all": float(visual_hybrid_metrics_all.get("ndcg_at_10", 0.0)),
        "visual_query_count_all": int(len(visual_query_ids)),
    }

    retrieval_metrics = evaluate_variants(
        eval_test_df, ["bm25_score", "dense_score", "hybrid_score"],
        k_values=(5, 10), positive_label_min=positive_label_min, ground_truth=test_truth,
    )
    ablation_metrics = evaluate_variants(
        eval_test_df, ["bm25_score", "dense_score", "hybrid_score", "ranker_score"],
        k_values=(5, 10), positive_label_min=positive_label_min, ground_truth=test_truth,
    )
    ranker_metrics_10 = metrics_at_k(
        eval_test_df, "ranker_score", k=10, positive_label_min=positive_label_min, ground_truth=test_truth
    )
    observed_ranker_metrics_10 = metrics_at_k(
        test_df, "ranker_score", k=10, positive_label_min=positive_label_min, ground_truth=observed_test_truth
    )
    ece = expected_calibration_error(
        test_df["calibrated_score"], eval_test_df["label"], positive_label_min=positive_label_min
    )
    brier = brier_score(
        test_df["calibrated_score"], eval_test_df["label"], positive_label_min=positive_label_min
    )
    observed_ece = expected_calibration_error(
        test_df["calibrated_score"], test_df["label"], positive_label_min=positive_label_min
    )
    observed_brier = brier_score(
        test_df["calibrated_score"], test_df["label"], positive_label_min=positive_label_min
    )
    slices = evaluate_slices(
        eval_test_df, test_truth, score_col="ranker_score", k=10, positive_label_min=positive_label_min,
        min_queries_for_claims=int(cfg.get("evaluation", {}).get("min_slice_queries_for_claims", 5)),
    )
    long_tail_row = slices[slices["slice"] == "long_tail"]
    cold_start_row = slices[slices["slice"] == "cold_start"]
    long_tail_recall = float(long_tail_row["recall_at_10"].iloc[0]) if len(long_tail_row) else 0.0
    cold_start_recall = float(cold_start_row["recall_at_10"].iloc[0]) if len(cold_start_row) else 0.0
    long_tail_recall_efficiency = float(long_tail_row["recall_efficiency_at_10"].iloc[0]) if len(long_tail_row) else 0.0
    cold_start_recall_efficiency = float(cold_start_row["recall_efficiency_at_10"].iloc[0]) if len(cold_start_row) else 0.0
    long_tail_hit_rate = float(long_tail_row["hit_rate_at_10"].iloc[0]) if len(long_tail_row) else 0.0
    cold_start_hit_rate = float(cold_start_row["hit_rate_at_10"].iloc[0]) if len(cold_start_row) else 0.0
    similar_to_self_return = _similar_to_self_return_rate(eval_test_df, score_col="ranker_score", k=10)
    query_type_metrics = {}
    for row in slices[slices["slice"].astype(str).str.startswith("query_type:")].to_dict("records"):
        query_type = str(row["slice"]).split(":", 1)[1]
        query_type_metrics[query_type] = {
            key: float(value) for key, value in row.items()
            if key in {"ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_efficiency_at_10", "precision_at_10", "hit_rate_at_10"}
            and pd.notna(value)
        }
        query_type_metrics[query_type]["num_queries"] = int(row.get("num_queries", 0))
        query_type_metrics[query_type]["claim_support"] = str(row.get("claim_support", "UNKNOWN"))

    latency_max_queries = int(cfg.get("evaluation", {}).get("latency_max_queries", 25))
    latency_warmup_queries = int(cfg.get("evaluation", {}).get("latency_warmup_queries", 2))
    sample_queries = queries["query"].head(max(1, latency_max_queries)).tolist()
    latency_kwargs = {"max_queries": latency_max_queries, "warmup_queries": latency_warmup_queries}
    bm25_latency, _ = measure_search_latency(
        lambda q, top_k=10: bm25.search(q, top_k=top_k), sample_queries, **latency_kwargs
    )
    dense_latency, _ = measure_search_latency(
        lambda q, top_k=10: vector_index.search(dense.encode_queries([q]), top_k=top_k), sample_queries, **latency_kwargs
    )
    hybrid_latency, _ = measure_search_latency(
        lambda q, top_k=10: hybrid.search(q, top_k=top_k), sample_queries, **latency_kwargs
    )
    ranker_latency, _ = measure_search_latency(
        lambda q, top_k=10: _search_with_ranker(q, catalog, bm25, dense, hybrid, ranker, top_k=top_k),
        sample_queries, **latency_kwargs
    )
    latency_by_variant = pd.DataFrame([
        {"variant": "bm25_score", **bm25_latency},
        {"variant": "dense_score", **dense_latency},
        {"variant": "hybrid_score", **hybrid_latency},
        {"variant": "ranker_score", **ranker_latency},
    ])

    hybrid_ndcg = float(ablation_metrics.loc[ablation_metrics["variant"] == "hybrid_score", "ndcg_at_10"].iloc[0])
    ranker_ndcg = float(ablation_metrics.loc[ablation_metrics["variant"] == "ranker_score", "ndcg_at_10"].iloc[0])
    ranker_lift = ranker_ndcg - hybrid_ndcg
    launch_metrics = dict(ranker_metrics_10)
    launch_metrics.update({
        "ece": ece,
        "brier": brier,
        "p95_latency_ms": ranker_latency["p95_latency_ms"],
        "long_tail_recall_at_10": long_tail_recall,
        "cold_start_recall_at_10": cold_start_recall,
        "long_tail_recall_efficiency_at_10": long_tail_recall_efficiency,
        "cold_start_recall_efficiency_at_10": cold_start_recall_efficiency,
        "long_tail_hit_rate_at_10": long_tail_hit_rate,
        "cold_start_hit_rate_at_10": cold_start_hit_rate,
        "similar_to_self_return_at_10": similar_to_self_return,
        "ranker_ndcg_lift_vs_hybrid": ranker_lift,
        "ranker_backend": ranker.backend,
        "dense_backend": dense.actual_backend,
        "vector_index_backend": vector_index.backend,
    })
    launch = evaluate_launch_gate(launch_metrics, cfg.get("launch_gates", {}))

    data_diagnostics = {
        "catalog_size": int(len(catalog)),
        "interaction_count": int(len(ratings)),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_std": float(ratings["rating"].std()),
        "item_popularity_gini": _gini(catalog["rating_count"]),
        "tag_coverage": float((catalog["tag_text"].astype(str).str.len() > 0).mean()),
        "catalog_cold_start_fraction": float(catalog["cold_start_flag"].mean()),
        "preference_rating_correlation": float(ratings[["preference_affinity", "rating"]].corr().iloc[0, 1]) if "preference_affinity" in ratings and ratings["preference_affinity"].nunique() > 1 else None,
        "label_disagreement_rate": float((labels["label"].astype(int) != labels.get("clean_label", labels["label"]).astype(int)).mean()),
        "evaluation_truth": evaluation_label_col,
        "split_anchor_leakage_free": bool(split_diagnostics.get("anchor_leakage_free", False)),
        "split_personalized_user_leakage_free": bool(split_diagnostics.get("personalized_user_leakage_free", False)),
        "enrichment": enrichment_diagnostics,
    }

    log("saving artifacts and reports")
    save_table(retrieval_metrics, out_dir / "retrieval_metrics.csv")
    save_table(ablation_metrics, out_dir / "ablation_metrics.csv")
    save_table(slices, out_dir / "slice_metrics.csv")
    save_table(test_df, out_dir / "test_predictions.csv")
    save_table(latency_by_variant, out_dir / "latency_by_variant.csv")
    save_table(test_candidate_diagnostics, out_dir / "candidate_recall_diagnostics.csv")
    write_json(out_dir / "latency_summary.json", ranker_latency)
    write_json(out_dir / "launch_gate.json", launch)
    joblib.dump({"ranker": ranker, "calibrator": calibrator, "feature_columns": FEATURE_COLUMNS}, out_dir / "ranker_bundle.joblib")
    # Embed the compact catalog so the zipped demo remains portable after extraction.
    joblib.dump({
        "bm25": bm25, "dense": dense, "vector_index": vector_index,
        "vector_index_backend": vector_index.backend,
        "catalog": catalog,
        "user_context": user_context,
        "catalog_path": "data/processed/media_catalog.csv",
    }, out_dir / "retrieval_bundle.joblib")

    write_report(reports_dir / "02_retrieval_baseline_report.md", "Retrieval Baseline Report", {
        "Data source": data_source,
        "Dense backend": dense.actual_backend,
        "Vector index backend": vector_index.backend,
        "Leakage control": "Validation/test metrics use retrieval-only candidates; positive-label injection is training-only.",
        "Metrics": _md_table(retrieval_metrics),
    })
    write_report(reports_dir / "03_lambdarank_training_report.md", "LambdaRank Training Report", {
        "Ranker backend": ranker.backend,
        "Feature columns": ", ".join(FEATURE_COLUMNS),
        "Train/validation/test queries": f"{len(train_q)} / {len(val_q)} / {len(test_q)}",
        "Training-only injected rows": str(int(train_df.get("training_label_injected", pd.Series(dtype=int)).sum())),
        "Calibration method": calibrator.actual_method,
        "Split strategy": split_diagnostics.get("strategy"),
        "Anchor leakage free": split_diagnostics.get("anchor_leakage_free"),
        "Personalized-user leakage free": split_diagnostics.get("personalized_user_leakage_free"),
    })
    write_report(reports_dir / "04_ablation_report.md", "Ablation Report", {
        "Model ladder": "BM25 → dense retrieval → hybrid retrieval → LambdaRank. Calibration changes confidence interpretation, not candidate generation.",
        "Metrics": _md_table(ablation_metrics),
        "Ranker NDCG@10 lift vs hybrid": f"{ranker_lift:+.4f}",
    })
    frontier = ablation_metrics.merge(latency_by_variant, on="variant", how="left")
    write_report(reports_dir / "05_quality_latency_frontier.md", "Quality-Latency Frontier", {
        "Per-variant latency": _md_table(latency_by_variant),
        "Quality-latency table": _md_table(frontier),
        "Interpretation": "The expensive learned ranker operates only after candidate retrieval. Calibrated score is reported for confidence and launch gating rather than as a separate retrieval model.",
    })
    write_report(reports_dir / "06_slice_reliability_report.md", "Slice Reliability Report", {
        "Methodology": "Query slices retain full candidate rankings. Long-tail/cold-start rows measure relevant sliced-item recall inside the full top-k list.",
        "Claim-support rule": f"Slices with fewer than {int(cfg.get('evaluation', {}).get('min_slice_queries_for_claims', 5))} held-out queries are marked LOW_SUPPORT and should not be used as standalone performance claims.",
        "Held-out slices": _md_table(slices),
        "Retrieval-only scenario diagnostic": json.dumps(diagnostic_metrics, indent=2),
    })
    write_report(reports_dir / "07_launch_readiness_memo.md", "Launch Readiness Memo", {
        "Decision": launch["decision"],
        "Checks": _md_table(pd.DataFrame(launch["checks"])),
        "Recommendation": "PASS means configured offline thresholds were met. Monte Carlo stability is assessed separately and should be reviewed before resume claims.",
    })
    write_model_card(reports_dir / "08_model_card.md", ranker.backend, dense.actual_backend, launch["decision"])
    write_claim_boundary(reports_dir / "09_claim_boundary.md")
    write_report(reports_dir / "16_split_integrity_report.md", "Split Integrity Report", {
        "Strategy": split_diagnostics.get("strategy"),
        "Split sizes": json.dumps(split_diagnostics.get("split_sizes", {}), indent=2),
        "Anchor overlap": json.dumps(split_diagnostics.get("anchor_overlap", {}), indent=2),
        "Personalized-user overlap": json.dumps(split_diagnostics.get("personalized_user_overlap", {}), indent=2),
        "Query-type counts": json.dumps(split_diagnostics.get("query_type_counts", {}), indent=2),
        "Interpretation": "Anchor-linked queries and repeated personalized users are assigned to one split to reduce train/validation/test leakage.",
    })
    observed_vs_evaluation = pd.DataFrame([
        {"truth_basis": "observed_label", **observed_ranker_metrics_10, "ece": observed_ece, "brier": observed_brier},
        {"truth_basis": evaluation_label_col, **ranker_metrics_10, "ece": ece, "brier": brier},
    ])
    write_report(reports_dir / "17_evaluation_truth_report.md", "Evaluation Truth Report", {
        "Training/calibration labels": "Observed proxy labels",
        "Primary evaluation truth": evaluation_label_col,
        "Observed vs primary metrics": _md_table(observed_vs_evaluation),
        "Interpretation": "Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.",
    })

    write_report(reports_dir / "28_candidate_recall_diagnostics.md", "Candidate Recall Diagnostics", {
        "Purpose": "Separates candidate-generation misses from reranker ordering errors on the frozen held-out benchmark.",
        "Metrics": _md_table(test_candidate_diagnostics),
        "Interpretation": "Improve candidate sources when recall efficiency is low; tune LambdaRank only when relevant items are already present in the pool.",
    })

    write_report(reports_dir / "26_metadata_enrichment_report.md", "Metadata Enrichment Report", {
        "Diagnostics": json.dumps(enrichment_diagnostics, indent=2),
        "Interpretation": "Tag Genome and IMDb enrichments are optional local adapters. Coverage and mapping method must be reviewed before performance claims.",
    })

    summary = {
        "data_source": data_source,
        "dense_backend": dense.actual_backend,
        "vector_index_backend": vector_index.backend,
        "ranker_backend": ranker.backend,
        "calibration_method": calibrator.actual_method,
        "evaluation_truth": evaluation_label_col,
        "metrics": launch_metrics,
        "observed_metrics": {**observed_ranker_metrics_10, "ece": observed_ece, "brier": observed_brier},
        "data_diagnostics": data_diagnostics,
        "variant_ndcg_at_10": {str(r.variant): float(r.ndcg_at_10) for r in ablation_metrics.itertuples()},
        "ranker_ndcg_lift_vs_hybrid": ranker_lift,
        "query_type_metrics": query_type_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "enrichment_diagnostics": enrichment_diagnostics,
        "frozen_benchmark_manifest": manifest_metadata,
        "candidate_diagnostics_artifact": str(out_dir / "candidate_recall_diagnostics.csv"),
        "similar_to_self_return_at_10": similar_to_self_return,
        "split_diagnostics": split_diagnostics,
        "launch_decision": launch["decision"],
        "reports_dir": str(reports_dir),
        "artifacts_dir": str(out_dir),
    }
    write_json(out_dir / "eval_summary.json", summary)
    log("done")
    return summary
