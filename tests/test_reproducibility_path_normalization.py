from scripts.reproducibility_check import _stable_summary


def test_stable_summary_normalizes_run_specific_artifact_path() -> None:
    left = {
        "candidate_diagnostics_artifact": (
            "temporary/run_a/artifacts/"
            "candidate_recall_diagnostics.csv"
        ),
        "metrics": {
            "ndcg_at_10": 0.42,
            "p95_latency_ms": 10.0,
        },
    }

    right = {
        "candidate_diagnostics_artifact": (
            "temporary/run_b/artifacts/"
            "candidate_recall_diagnostics.csv"
        ),
        "metrics": {
            "ndcg_at_10": 0.42,
            "p95_latency_ms": 20.0,
        },
    }

    normalized_left = _stable_summary(left)
    normalized_right = _stable_summary(right)

    assert normalized_left == normalized_right
    assert normalized_left[
        "candidate_diagnostics_artifact"
    ] == "candidate_recall_diagnostics.csv"
