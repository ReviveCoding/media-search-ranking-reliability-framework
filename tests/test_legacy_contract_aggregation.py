from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNS = (
    "core_conservative",
    "core_balanced",
    "core_compact",
    "core_champion_replay",
    "tag_genome_feature_only",
    "imdb_feature_only",
    "combined_feature_only",
)


def _summary(ndcg: float, *, split: str, tag: bool = False, imdb: bool = False) -> dict:
    q = {
        name: {"ndcg_at_10": value}
        for name, value in {
            "genre_tag": 0.50,
            "mood_decade": 0.10,
            "personalized": 0.20,
            "similar_to": 0.12,
            "visual_query": 0.70,
        }.items()
    }
    return {
        "metrics": {
            "ndcg_at_10": ndcg,
            "mrr_at_10": 0.5,
            "recall_efficiency_at_10": 0.25,
            "hit_rate_at_10": 0.6,
            "ece": 0.01,
            "p95_latency_ms": 200.0,
        },
        "ranker_ndcg_lift_vs_hybrid": 0.18,
        "query_type_metrics": q,
        "enrichment_diagnostics": {
            "tag_genome_applied": tag,
            "imdb_applied": imdb,
            "retrieval_text_changed": False,
        },
        "frozen_benchmark_manifest": {"version": 1, "query_count": 500},
        "split_diagnostics": {"strategy": split, "split_sizes": {"test": 99}},
        "launch_decision": "ITERATE",
    }


def _write_run(root: Path, name: str, summary: dict, fingerprint: str | None = None) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if fingerprint is not None:
        (run_dir / "run_config_sha256.txt").write_text(fingerprint, encoding="utf-8")


def test_legacy_split_drift_does_not_block_aggregation(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    script = Path(__file__).parents[1] / "scripts" / "aggregate_frozen_movielens_results.py"

    _write_run(output, "core_conservative", _summary(0.28, split="legacy-a"))
    _write_run(output, "core_balanced", _summary(0.29, split="legacy-b"))
    _write_run(output, "core_compact", _summary(0.27, split="legacy-c"))
    _write_run(output, "core_champion_replay", _summary(0.30, split="frozen"), "new")
    _write_run(output, "tag_genome_feature_only", _summary(0.31, split="frozen", tag=True), "tag")
    _write_run(output, "imdb_feature_only", _summary(0.305, split="frozen", imdb=True), "imdb")
    _write_run(output, "combined_feature_only", _summary(0.32, split="frozen", tag=True, imdb=True), "both")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(output),
            "--reports-root",
            str(reports),
            "--manifest-dir",
            str(tmp_path / "manifest"),
        ],
        check=True,
    )

    result = json.loads((output / "frozen_ablation_summary.json").read_text(encoding="utf-8"))
    assert result["champion"]["run_name"] == "combined_feature_only"
    assert result["replay_diagnostics_status"] == "legacy_not_strictly_comparable"
    assert (output / "frozen_ablation_results.csv").is_file()


def test_strict_enrichment_split_drift_still_fails(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    script = Path(__file__).parents[1] / "scripts" / "aggregate_frozen_movielens_results.py"

    for name in ("core_conservative", "core_balanced", "core_compact"):
        _write_run(output, name, _summary(0.28, split=f"legacy-{name}"))
    _write_run(output, "core_champion_replay", _summary(0.30, split="frozen"), "new")
    _write_run(output, "tag_genome_feature_only", _summary(0.31, split="wrong", tag=True), "tag")
    _write_run(output, "imdb_feature_only", _summary(0.305, split="frozen", imdb=True), "imdb")
    _write_run(output, "combined_feature_only", _summary(0.32, split="frozen", tag=True, imdb=True), "both")

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(output),
            "--reports-root",
            str(reports),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "tag_genome_feature_only used different query-group splits" in proc.stderr
