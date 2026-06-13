from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/frozen_quality_ablation"
BENCHMARK = ROOT / "data/benchmarks/ml10m_frozen_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_query() -> str:
    path = BENCHMARK / "queries.csv"
    if not path.is_file():
        return "benchmark query unavailable"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        return "benchmark query unavailable"
    for key in ("query", "query_text", "text", "prompt"):
        if row.get(key):
            return row[key]
    return next((value for value in row.values() if value), "benchmark query unavailable")


def main() -> None:
    summary = load_json(ARTIFACTS / "frozen_ablation_summary.json")
    diagnostics = load_json(ARTIFACTS / "frozen_replay_diagnostics.json")
    champion = summary["champion"]
    baseline = summary["canonical_core_baseline"]
    b = float(baseline["ndcg_at_10"])
    c = float(champion["ndcg_at_10"])

    print("MEDIA SEARCH RELIABILITY: PUBLIC DEMO")
    print("=" * 44)
    print(f"sample_query={sample_query()}")
    print("flow=query_understanding -> hybrid_retrieval -> enrichment -> lambdarank")
    print("evaluation=ranking_metrics + slices + calibration + latency + launch_gate")
    print(f"champion={champion['run_name']}")
    print(f"canonical_baseline={baseline['run_name']}")
    print(f"baseline_ndcg_at_10={b:.6f}")
    print(f"champion_ndcg_at_10={c:.6f}")
    print(f"relative_ndcg_change_pct={100.0 * (c - b) / b:+.2f}")
    print(f"canonical_contract_validated={diagnostics['canonical_contract_validated']}")
    print(f"strict_replay_claim={diagnostics['strict_replay_claim']}")
    print(f"launch_decision={champion.get('launch_decision', 'UNKNOWN')}")


if __name__ == "__main__":
    main()
