from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_REPORTS = [
    "01_data_validation_report.md",
    "02_retrieval_baseline_report.md",
    "03_lambdarank_training_report.md",
    "04_ablation_report.md",
    "05_quality_latency_frontier.md",
    "06_slice_reliability_report.md",
    "07_launch_readiness_memo.md",
    "08_model_card.md",
    "09_claim_boundary.md",
    "10_query_label_quality_report.md",
    "16_split_integrity_report.md",
    "17_evaluation_truth_report.md",
]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unit tests, demo pipeline, and artifact checks.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--mode", choices=["demo", "movielens"], default="demo")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run([sys.executable, "scripts/run_pipeline.py", "--config", args.config, "--mode", args.mode])

    summary_path = root / "artifacts" / "eval_summary.json"
    if not summary_path.exists():
        raise SystemExit("Missing artifacts/eval_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("launch_decision") == "BLOCK":
        raise SystemExit("Launch decision is BLOCK")

    missing = [r for r in REQUIRED_REPORTS if not (root / "reports" / r).exists()]
    if missing:
        raise SystemExit(f"Missing reports: {missing}")

    print("\nSmoke test PASS")
    print(json.dumps({
        "launch_decision": summary.get("launch_decision"),
        "ranker_backend": summary.get("ranker_backend"),
        "dense_backend": summary.get("dense_backend"),
        "vector_index_backend": summary.get("vector_index_backend"),
        "metrics": summary.get("metrics"),
    }, indent=2))


if __name__ == "__main__":
    main()
