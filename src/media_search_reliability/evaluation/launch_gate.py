from __future__ import annotations


def evaluate_launch_gate(metrics: dict, gates: dict) -> dict:
    checks = []

    def add(name, passed, value, threshold):
        checks.append({"check": name, "passed": bool(passed), "value": value, "threshold": threshold})

    add("ndcg_at_10", metrics.get("ndcg_at_10", 0) >= gates.get("ndcg_at_10_min", 0), metrics.get("ndcg_at_10", 0), gates.get("ndcg_at_10_min", 0))
    if "recall_at_10_min" in gates:
        add("recall_at_10", metrics.get("recall_at_10", 0) >= gates["recall_at_10_min"], metrics.get("recall_at_10", 0), gates["recall_at_10_min"])
    add("recall_efficiency_at_10", metrics.get("recall_efficiency_at_10", 0) >= gates.get("recall_efficiency_at_10_min", 0), metrics.get("recall_efficiency_at_10", 0), gates.get("recall_efficiency_at_10_min", 0))
    add("mrr_at_10", metrics.get("mrr_at_10", 0) >= gates.get("mrr_at_10_min", 0), metrics.get("mrr_at_10", 0), gates.get("mrr_at_10_min", 0))
    add("ece", metrics.get("ece", 1) <= gates.get("ece_max", 1), metrics.get("ece", 1), gates.get("ece_max", 1))
    add("p95_latency_ms", metrics.get("p95_latency_ms", 999999) <= gates.get("p95_latency_ms_max", 999999), metrics.get("p95_latency_ms", 999999), gates.get("p95_latency_ms_max", 999999))
    if "long_tail_recall_min" in gates:
        add("long_tail_recall", metrics.get("long_tail_recall_at_10", 0) >= gates["long_tail_recall_min"], metrics.get("long_tail_recall_at_10", 0), gates["long_tail_recall_min"])
    if "cold_start_recall_min" in gates:
        add("cold_start_recall", metrics.get("cold_start_recall_at_10", 0) >= gates["cold_start_recall_min"], metrics.get("cold_start_recall_at_10", 0), gates["cold_start_recall_min"])
    if "long_tail_recall_efficiency_min" in gates:
        add("long_tail_recall_efficiency", metrics.get("long_tail_recall_efficiency_at_10", 0) >= gates["long_tail_recall_efficiency_min"], metrics.get("long_tail_recall_efficiency_at_10", 0), gates["long_tail_recall_efficiency_min"])
    if "cold_start_recall_efficiency_min" in gates:
        add("cold_start_recall_efficiency", metrics.get("cold_start_recall_efficiency_at_10", 0) >= gates["cold_start_recall_efficiency_min"], metrics.get("cold_start_recall_efficiency_at_10", 0), gates["cold_start_recall_efficiency_min"])
    if "long_tail_hit_rate_min" in gates:
        add("long_tail_hit_rate", metrics.get("long_tail_hit_rate_at_10", 0) >= gates["long_tail_hit_rate_min"], metrics.get("long_tail_hit_rate_at_10", 0), gates["long_tail_hit_rate_min"])
    if "cold_start_hit_rate_min" in gates:
        add("cold_start_hit_rate", metrics.get("cold_start_hit_rate_at_10", 0) >= gates["cold_start_hit_rate_min"], metrics.get("cold_start_hit_rate_at_10", 0), gates["cold_start_hit_rate_min"])
    if "ranker_ndcg_lift_min" in gates:
        add("ranker_ndcg_lift", metrics.get("ranker_ndcg_lift_vs_hybrid", -999) >= gates["ranker_ndcg_lift_min"], metrics.get("ranker_ndcg_lift_vs_hybrid", -999), gates["ranker_ndcg_lift_min"])
    if "similar_to_self_return_max" in gates:
        add("similar_to_self_return", metrics.get("similar_to_self_return_at_10", 1) <= gates["similar_to_self_return_max"], metrics.get("similar_to_self_return_at_10", 1), gates["similar_to_self_return_max"])

    required_backend = gates.get("required_ranker_backend_prefix")
    if required_backend:
        actual_backend = str(metrics.get("ranker_backend", ""))
        add("ranker_backend_contract", actual_backend.startswith(str(required_backend)), actual_backend, f"prefix:{required_backend}")

    required_dense = gates.get("required_dense_backend_prefix")
    if required_dense:
        actual_dense = str(metrics.get("dense_backend", ""))
        add("dense_backend_contract", actual_dense.startswith(str(required_dense)), actual_dense, f"prefix:{required_dense}")
    required_vector = gates.get("required_vector_index_backend_prefix")
    if required_vector:
        actual_vector = str(metrics.get("vector_index_backend", ""))
        add("vector_index_backend_contract", actual_vector.startswith(str(required_vector)), actual_vector, f"prefix:{required_vector}")

    approved = gates.get("approved_public_or_synthetic_data_only", gates.get("public_dataset_only", True))
    add("approved_public_or_synthetic_data_only", approved, approved, True)
    add("no_private_user_data", gates.get("no_private_user_data", True), gates.get("no_private_user_data", True), True)
    add("no_production_deployment_claim", gates.get("no_production_deployment_claim", True), gates.get("no_production_deployment_claim", True), True)

    failed = [check for check in checks if not check["passed"]]
    hard_boundaries = {"approved_public_or_synthetic_data_only", "no_private_user_data", "no_production_deployment_claim"}
    if not failed:
        decision = "PASS"
    elif any(check["check"] in hard_boundaries for check in failed):
        decision = "BLOCK"
    elif len(failed) <= 2:
        decision = "REVIEW"
    else:
        decision = "ITERATE"
    return {"decision": decision, "checks": checks}
