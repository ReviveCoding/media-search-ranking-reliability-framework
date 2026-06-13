from pathlib import Path
import yaml

from media_search_reliability.pipeline import run_pipeline


def test_synthetic_pipeline_uses_clean_truth_under_label_noise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "project": {"random_seed": 7, "output_dir": "artifacts", "reports_dir": "reports", "data_dir": "data"},
        "data": {
            "demo_movies": 70, "demo_users": 45, "demo_ratings": 700,
            "synthetic": {
                "popularity_alpha": 1.05, "preference_strength": 1.2, "rating_noise": 0.9,
                "tag_observation_prob": 0.35, "exploration_rate": 0.12, "cold_start_fraction": 0.08,
            },
        },
        "queries": {
            "num_queries": 30, "candidates_per_query": 20, "label_all_catalog_demo": True,
            "label_all_catalog_movielens": False, "label_noise": 0.2,
        },
        "retrieval": {
            "top_k": 20, "hybrid_alpha": 0.55, "dense_backend": "tfidf",
            "dense_model_name": "sentence-transformers/all-MiniLM-L6-v2", "dense_device": "cpu",
            "dense_batch_size": 32, "faiss_use_gpu_if_available": False,
        },
        "ranking": {
            "test_size": 0.2, "val_size": 0.15, "objective": "lambdarank", "metric": "ndcg",
            "n_estimators": 20, "learning_rate": 0.05, "num_leaves": 15,
            "min_child_samples": 5, "device_type": "cpu", "n_jobs": 1,
        },
        "evaluation": {"positive_label_min": 2, "latency_max_queries": 1, "latency_warmup_queries": 0},
        "calibration": {"method": "platt", "positive_label_min": 2},
        "launch_gates": {
            "ndcg_at_10_min": 0.0, "recall_efficiency_at_10_min": 0.0, "mrr_at_10_min": 0.0,
            "ece_max": 1.0, "p95_latency_ms_max": 9999, "long_tail_recall_min": 0.0,
            "cold_start_recall_min": 0.0, "ranker_ndcg_lift_min": -1.0,
            "similar_to_self_return_max": 1.0, "approved_public_or_synthetic_data_only": True,
            "no_private_user_data": True, "no_production_deployment_claim": True,
        },
    }
    config_path = Path("pipeline.yaml")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = run_pipeline(config_path, mode="demo")
    assert summary["evaluation_truth"] == "clean_label"
    assert summary["data_diagnostics"]["label_disagreement_rate"] > 0
    assert "observed_metrics" in summary
