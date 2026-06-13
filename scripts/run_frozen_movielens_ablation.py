from __future__ import annotations

import argparse
import copy
import json
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml



RANKER_PROFILES = {
    "conservative": {
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 450,
        "min_child_samples": 30,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "early_stopping_rounds": 50,
        "label_gain": [0, 1, 3, 7],
    },
    "balanced": {
        "num_leaves": 63,
        "learning_rate": 0.03,
        "n_estimators": 550,
        "min_child_samples": 20,
        "reg_lambda": 0.5,
        "reg_alpha": 0.05,
        "feature_fraction": 0.95,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "early_stopping_rounds": 60,
        "label_gain": [0, 1, 3, 7],
    },
    "compact": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 280,
        "min_child_samples": 20,
        "reg_lambda": 0.5,
        "reg_alpha": 0.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "early_stopping_rounds": 40,
        "label_gain": [0, 1, 3, 7],
    },
}


REPLAY_METRICS = ("ndcg_at_10", "mrr_at_10", "recall_efficiency_at_10")
DEFAULT_REPLAY_ATOL = 1e-3
DEFAULT_REPLAY_RTOL = 1e-3


def _replay_diagnostics(core_original: pd.Series, core_replay: pd.Series, *, atol: float, rtol: float) -> dict:
    metrics: dict[str, dict] = {}
    passed = True
    for name in REPLAY_METRICS:
        original = float(core_original[name])
        replay = float(core_replay[name])
        delta = replay - original
        close = math.isclose(replay, original, rel_tol=rtol, abs_tol=atol)
        metrics[name] = {
            "original": original,
            "replay": replay,
            "delta": delta,
            "absolute_delta": abs(delta),
            "abs_tolerance": atol,
            "rel_tolerance": rtol,
            "passed": close,
        }
        passed = passed and close
    return {"passed": passed, "metrics": metrics}


CANDIDATE_QUOTAS = {
    "default": {"bm25": 50, "dense": 50, "hybrid": 60, "specialized": 30},
    "similar_to": {"bm25": 20, "dense": 30, "hybrid": 40, "specialized": 90},
    "personalized": {"bm25": 20, "dense": 30, "hybrid": 40, "specialized": 90},
    "mood_decade": {"bm25": 60, "dense": 60, "hybrid": 70, "specialized": 0},
    "genre_tag": {"bm25": 60, "dense": 50, "hybrid": 70, "specialized": 0},
    "visual_query": {"bm25": 35, "dense": 70, "hybrid": 70, "specialized": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen MovieLens quality and metadata ablation")
    parser.add_argument("--config", default="configs/pipeline_external_gpu_quick.yaml")
    parser.add_argument("--movielens-dir", required=True)
    parser.add_argument("--tag-genome-enrichment")
    parser.add_argument("--imdb-enrichment")
    parser.add_argument("--manifest-dir", default="data/benchmarks/ml10m_frozen_v1")
    parser.add_argument("--output-root", default="artifacts/frozen_quality_ablation")
    parser.add_argument("--reports-root", default="reports/frozen_quality_ablation")
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--candidates-per-query", type=int, default=100)
    parser.add_argument("--skip-ranker-tuning", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed variant summaries when their generated config fingerprint matches.",
    )
    return parser.parse_args()


def _write_config(base: dict, path: Path, *, variant: str, output_root: Path, reports_root: Path,
                  manifest_dir: Path, manifest_mode: str, profile: dict,
                  tag_path: str | None, imdb_path: str | None, args: argparse.Namespace) -> None:
    cfg = copy.deepcopy(base)
    cfg.setdefault("project", {})["output_dir"] = str((output_root / variant).resolve())
    cfg["project"]["reports_dir"] = str((reports_root / variant).resolve())
    cfg.setdefault("data", {})["movielens_raw_dir"] = str(Path(args.movielens_dir).resolve())
    cfg.setdefault("queries", {})["num_queries"] = int(args.num_queries)
    cfg["queries"]["candidates_per_query"] = int(args.candidates_per_query)
    cfg.setdefault("retrieval", {})["top_k"] = 80
    cfg["retrieval"]["candidate_quotas"] = CANDIDATE_QUOTAS
    cfg.setdefault("ranking", {}).update(profile)
    cfg["ranking"]["eval_at"] = [5, 10]
    cfg.setdefault("benchmark", {}).update({
        "manifest_dir": str(manifest_dir.resolve()),
        "manifest_mode": manifest_mode,
        "require_catalog_fingerprint": True,
    })
    cfg["enrichment"] = {
        "mode": "feature_only",
        "tag_genome_enrichment_path": tag_path,
        "imdb_enrichment_path": imdb_path,
        "top_text_tags": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_variant(base: dict, temp_dir: Path, *, name: str, output_root: Path, reports_root: Path,
                 manifest_dir: Path, manifest_mode: str, profile: dict,
                 tag_path: str | None, imdb_path: str | None, args: argparse.Namespace) -> dict:
    cfg_path = temp_dir / f"{name}.yaml"
    _write_config(
        base, cfg_path, variant=name, output_root=output_root, reports_root=reports_root,
        manifest_dir=manifest_dir, manifest_mode=manifest_mode, profile=profile,
        tag_path=tag_path, imdb_path=imdb_path, args=args,
    )

    variant_dir = (output_root / name).resolve()
    summary_path = variant_dir / "eval_summary.json"
    fingerprint_path = variant_dir / "run_config_sha256.txt"
    config_fingerprint = _sha256(cfg_path)

    if args.resume and summary_path.is_file():
        stored_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip() if fingerprint_path.is_file() else None
        # v10 results created before this hotfix have no sidecar. They are safe to reuse
        # only when the user explicitly requested --resume. New runs always persist one.
        if stored_fingerprint is None or stored_fingerprint == config_fingerprint:
            print(f"[resume] reusing completed variant: {name}", flush=True)
            return json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[resume] config changed; rerunning variant: {name}", flush=True)

    env = os.environ.copy()
    env.pop("TAG_GENOME_ENRICHMENT_PATH", None)
    env.pop("IMDB_ENRICHMENT_PATH", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    runner = Path(__file__).resolve().with_name("run_pipeline.py")
    command = [sys.executable, str(runner), "--config", str(cfg_path.resolve()), "--mode", "movielens"]
    print(f"[variant] starting isolated process: {name}", flush=True)
    subprocess.run(command, check=True, cwd=Path.cwd(), env=env)

    if not summary_path.is_file():
        raise FileNotFoundError(f"Variant completed without eval_summary.json: {summary_path}")
    fingerprint_path.write_text(config_fingerprint + "\n", encoding="utf-8")
    print(f"[variant] completed and released process memory: {name}", flush=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _row(name: str, summary: dict, *, profile: str, variant: str) -> dict:
    metrics = summary["metrics"]
    q = summary.get("query_type_metrics", {})
    return {
        "run_name": name,
        "variant": variant,
        "ranker_profile": profile,
        "ndcg_at_10": metrics.get("ndcg_at_10"),
        "mrr_at_10": metrics.get("mrr_at_10"),
        "recall_efficiency_at_10": metrics.get("recall_efficiency_at_10"),
        "hit_rate_at_10": metrics.get("hit_rate_at_10"),
        "ranker_lift_vs_hybrid": summary.get("ranker_ndcg_lift_vs_hybrid"),
        "genre_tag_ndcg": q.get("genre_tag", {}).get("ndcg_at_10"),
        "mood_decade_ndcg": q.get("mood_decade", {}).get("ndcg_at_10"),
        "personalized_ndcg": q.get("personalized", {}).get("ndcg_at_10"),
        "similar_to_ndcg": q.get("similar_to", {}).get("ndcg_at_10"),
        "visual_query_ndcg": q.get("visual_query", {}).get("ndcg_at_10"),
        "ece": metrics.get("ece"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "launch_decision": summary.get("launch_decision"),
        "tag_genome_applied": summary.get("enrichment_diagnostics", {}).get("tag_genome_applied", False),
        "imdb_applied": summary.get("enrichment_diagnostics", {}).get("imdb_applied", False),
        "retrieval_text_changed": summary.get("enrichment_diagnostics", {}).get("retrieval_text_changed", False),
    }


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    manifest_dir = Path(args.manifest_dir)
    temp_dir = output_root / "_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    summaries: dict[str, dict] = {}
    profiles = {"compact": RANKER_PROFILES["compact"]} if args.skip_ranker_tuning else RANKER_PROFILES

    manifest_exists = (manifest_dir / "manifest.json").is_file()
    for profile_name, profile in profiles.items():
        run_name = f"core_{profile_name}"
        summary = _run_variant(
            base, temp_dir, name=run_name, output_root=output_root, reports_root=reports_root,
            manifest_dir=manifest_dir, manifest_mode="reuse" if manifest_exists else "create",
            profile=profile, tag_path=None, imdb_path=None, args=args,
        )
        manifest_exists = True
        summaries[run_name] = summary
        results.append(_row(run_name, summary, profile=profile_name, variant="core"))

    table = pd.DataFrame(results)
    champion_row = table.sort_values(["ndcg_at_10", "recall_efficiency_at_10"], ascending=False).iloc[0]
    champion_profile = str(champion_row["ranker_profile"])
    champion_config = RANKER_PROFILES[champion_profile]

    metadata_variants = [
        ("core_champion_replay", None, None),
        ("tag_genome_feature_only", args.tag_genome_enrichment, None),
        ("imdb_feature_only", None, args.imdb_enrichment),
        ("combined_feature_only", args.tag_genome_enrichment, args.imdb_enrichment),
    ]
    for name, tag_path, imdb_path in metadata_variants:
        if name == "tag_genome_feature_only" and not tag_path:
            continue
        if name == "imdb_feature_only" and not imdb_path:
            continue
        if name == "combined_feature_only" and not (tag_path and imdb_path):
            continue
        summary = _run_variant(
            base, temp_dir, name=name, output_root=output_root, reports_root=reports_root,
            manifest_dir=manifest_dir, manifest_mode="reuse", profile=champion_config,
            tag_path=tag_path, imdb_path=imdb_path, args=args,
        )
        summaries[name] = summary
        results.append(_row(name, summary, profile=champion_profile, variant=name))

    final = pd.DataFrame(results)
    core_replay = final[final["run_name"] == "core_champion_replay"].iloc[0]
    core_original = final[final["run_name"] == f"core_{champion_profile}"].iloc[0]

    original_summary = summaries[f"core_{champion_profile}"]
    replay_summary = summaries["core_champion_replay"]
    if original_summary.get("frozen_benchmark_manifest") != replay_summary.get("frozen_benchmark_manifest"):
        raise AssertionError("Frozen core replay used a different benchmark manifest")
    if original_summary.get("split_diagnostics") != replay_summary.get("split_diagnostics"):
        raise AssertionError("Frozen core replay used different query-group splits")

    replay_atol = float(os.environ.get("FROZEN_REPLAY_ATOL", DEFAULT_REPLAY_ATOL))
    replay_rtol = float(os.environ.get("FROZEN_REPLAY_RTOL", DEFAULT_REPLAY_RTOL))
    replay_check = _replay_diagnostics(core_original, core_replay, atol=replay_atol, rtol=replay_rtol)
    replay_check.update({
        "core_profile": champion_profile,
        "original_run": f"core_{champion_profile}",
        "replay_run": "core_champion_replay",
        "original_ranker_backend": original_summary.get("metrics", {}).get("ranker_backend"),
        "replay_ranker_backend": replay_summary.get("metrics", {}).get("ranker_backend"),
        "manifest_equal": True,
        "split_equal": True,
    })
    (output_root / "frozen_replay_diagnostics.json").write_text(
        json.dumps(replay_check, indent=2, default=str), encoding="utf-8"
    )
    if not replay_check["passed"]:
        failed = [name for name, detail in replay_check["metrics"].items() if not detail["passed"]]
        raise AssertionError(
            "Frozen core replay exceeded GPU tolerance for: " + ", ".join(failed)
        )
    max_delta = max(detail["absolute_delta"] for detail in replay_check["metrics"].values())
    print(
        f"[replay] frozen manifest/splits match; GPU metric replay within tolerance "
        f"(max_abs_delta={max_delta:.8g}, atol={replay_atol}, rtol={replay_rtol})",
        flush=True,
    )

    slice_columns = ["genre_tag_ndcg", "mood_decade_ndcg", "personalized_ndcg", "similar_to_ndcg", "visual_query_ndcg"]
    def eligible(row: pd.Series) -> bool:
        if str(row["run_name"]) == "core_champion_replay":
            return True
        if str(row["variant"]) == "core":
            return False
        if bool(row.get("retrieval_text_changed", False)):
            return False
        if float(row.get("ranker_lift_vs_hybrid", 0.0) or 0.0) <= 0:
            return False
        for col in slice_columns:
            base_value = core_replay.get(col)
            new_value = row.get(col)
            if pd.notna(base_value) and float(base_value) > 0 and pd.notna(new_value):
                if float(new_value) < 0.90 * float(base_value):
                    return False
        return float(row["ndcg_at_10"]) >= float(core_replay["ndcg_at_10"])

    final["eligible_for_promotion"] = final.apply(eligible, axis=1)
    eligible_rows = final[final["eligible_for_promotion"]]
    champion_row_final = eligible_rows.sort_values(
        ["ndcg_at_10", "recall_efficiency_at_10"], ascending=False, kind="mergesort"
    ).iloc[0]
    final = final.sort_values("ndcg_at_10", ascending=False, kind="mergesort")
    final.to_csv(output_root / "frozen_ablation_results.csv", index=False)
    champion = champion_row_final.to_dict()
    (output_root / "frozen_ablation_summary.json").write_text(
        json.dumps({
            "champion": champion,
            "core_champion": core_replay.to_dict(),
            "manifest_dir": str(manifest_dir.resolve()),
            "promotion_rule": "no retrieval-text drift, positive ranker lift, no >10% slice regression, non-negative overall NDCG change",
        }, indent=2, default=str),
        encoding="utf-8",
    )

    md = [
        "# Frozen MovieLens Ranking Quality Ablation",
        "",
        "All runs reuse the same catalog fingerprint, queries, proxy judgments, and train/validation/test query groups.",
        "Tag Genome and IMDb operate in feature-only mode and do not rewrite retrieval text or evaluation truth.",
        "",
        final.to_markdown(index=False),
        "",
        f"## Champion\n\n`{champion['run_name']}` with NDCG@10 `{float(champion['ndcg_at_10']):.4f}`.",
        "",
        "## Interpretation",
        "",
        "Candidate recall diagnostics are stored inside each variant artifact directory. If candidate recall is low, improve retrieval routing; if recall is high but NDCG is low, tune the ranker.",
    ]
    (reports_root / "29_frozen_movielens_ablation.md").write_text("\n".join(md), encoding="utf-8")
    print(final.to_string(index=False))
    print(f"Champion: {champion['run_name']} NDCG@10={float(champion['ndcg_at_10']):.6f}")
    print(f"Report: {(reports_root / '29_frozen_movielens_ablation.md').resolve()}")


if __name__ == "__main__":
    main()
