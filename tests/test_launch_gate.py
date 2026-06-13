from media_search_reliability.evaluation.launch_gate import evaluate_launch_gate


def test_launch_gate_blocks_claim_boundary_violation():
    result = evaluate_launch_gate(
        {"ndcg_at_10": 1, "recall_at_10": 1, "mrr_at_10": 1, "ece": 0.0, "p95_latency_ms": 1, "long_tail_recall_at_10": 1},
        {"public_dataset_only": False, "no_private_user_data": True, "no_production_deployment_claim": True},
    )
    assert result["decision"] == "BLOCK"


def test_launch_gate_does_not_pass_non_ranking_backend_when_required():
    metrics = {
        "ndcg_at_10": 0.9, "recall_efficiency_at_10": 0.9, "mrr_at_10": 0.9,
        "ece": 0.01, "p95_latency_ms": 10, "ranker_backend": "sklearn-gbr-explicit-fallback",
    }
    gates = {
        "ndcg_at_10_min": 0.3, "recall_efficiency_at_10_min": 0.5, "mrr_at_10_min": 0.2,
        "ece_max": 0.18, "p95_latency_ms_max": 500,
        "required_ranker_backend_prefix": "lightgbm-lambdarank",
    }
    result = evaluate_launch_gate(metrics, gates)
    assert result["decision"] != "PASS"
    assert any(check["check"] == "ranker_backend_contract" and not check["passed"] for check in result["checks"])


def test_launch_gate_can_require_dense_and_vector_backends():
    metrics = {
        "ndcg_at_10": 0.9, "recall_efficiency_at_10": 0.9, "mrr_at_10": 0.9,
        "ece": 0.01, "p95_latency_ms": 10,
        "ranker_backend": "lightgbm-lambdarank-gpu",
        "dense_backend": "tfidf-normalized-fallback",
        "vector_index_backend": "sklearn-nearest-neighbors-fallback",
    }
    gates = {
        "ndcg_at_10_min": 0.3, "recall_efficiency_at_10_min": 0.5, "mrr_at_10_min": 0.2,
        "ece_max": 0.18, "p95_latency_ms_max": 500,
        "required_ranker_backend_prefix": "lightgbm-lambdarank",
        "required_dense_backend_prefix": "sentence-transformers",
        "required_vector_index_backend_prefix": "faiss",
    }
    result = evaluate_launch_gate(metrics, gates)
    failed = {check["check"] for check in result["checks"] if not check["passed"]}
    assert {"dense_backend_contract", "vector_index_backend_contract"}.issubset(failed)
    assert result["decision"] != "PASS"
