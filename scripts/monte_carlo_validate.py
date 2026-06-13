from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

# Keep repeated small trials stable on local machines and CI. Numerical libraries
# otherwise may create many worker threads per independent simulation.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

from media_search_reliability.pipeline import run_pipeline
from media_search_reliability.evaluation.monte_carlo import sample_scenario_parameters

ROOT = Path(__file__).resolve().parents[1]




def _quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else float("nan")


def _metric_bounds_ok(row: dict) -> bool:
    bounded = ["ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_efficiency_at_10", "precision_at_10", "hit_rate_at_10", "ece", "brier", "long_tail_recall_at_10", "cold_start_recall_at_10", "similar_to_self_return_at_10"]
    return all(np.isfinite(row.get(k, float("nan"))) and 0.0 <= float(row[k]) <= 1.0 for k in bounded)


def _scenario_summary(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_efficiency_at_10", "hit_rate_at_10", "ece", "brier",
        "p95_latency_ms", "long_tail_recall_at_10", "cold_start_recall_at_10",
        "ranker_ndcg_lift_vs_hybrid", "item_popularity_gini", "tag_coverage",
        "rating_mean", "rating_std", "preference_rating_correlation", "similar_to_self_return_at_10",
        "visual_query_ndcg_at_10", "visual_query_hybrid_ndcg_all", "mood_decade_ndcg_at_10", "label_disagreement_rate",
        "param_popularity_alpha", "param_preference_strength", "param_rating_noise",
        "param_tag_observation_prob", "param_exploration_rate", "param_cold_start_fraction", "param_label_noise",
    ]
    rows = []
    for scenario, group in trials.groupby("scenario"):
        row = {"scenario": scenario, "trials": len(group), "success_rate": float(group["success"].mean())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
            row[f"{metric}_p05"] = _quantile(values, 0.05)
            row[f"{metric}_p95"] = _quantile(values, 0.95)
        row["pass_or_review_rate"] = float(group["launch_decision"].isin(["PASS", "REVIEW"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)




def _paired_deltas(trials: pd.DataFrame, baseline: str, stressed: str, metric: str) -> pd.Series:
    """Return stressed minus baseline values paired by trial index."""
    cols = ["trial", metric]
    left = trials[trials["scenario"] == baseline][cols].rename(columns={metric: "baseline"})
    right = trials[trials["scenario"] == stressed][cols].rename(columns={metric: "stressed"})
    paired = left.merge(right, on="trial", how="inner")
    return pd.to_numeric(paired["stressed"], errors="coerce") - pd.to_numeric(paired["baseline"], errors="coerce")

def _checks(trials: pd.DataFrame, summary: pd.DataFrame, acceptance: dict) -> list[dict]:
    successful = trials[trials["success"]].copy()
    checks = []
    def add(name, passed, value, threshold, interpretation):
        checks.append({"check": name, "passed": bool(passed), "value": value, "threshold": threshold, "interpretation": interpretation})

    success_rate = float(trials["success"].mean()) if len(trials) else 0.0
    add("pipeline_success_rate", success_rate >= acceptance["pipeline_success_rate_min"], success_rate, acceptance["pipeline_success_rate_min"], "Every independent trial should complete end to end.")
    finite_rate = float(successful["metric_bounds_ok"].mean()) if len(successful) else 0.0
    add("finite_metric_rate", finite_rate >= acceptance["finite_metric_rate_min"], finite_rate, acceptance["finite_metric_rate_min"], "Metrics must be finite and lie in expected bounds.")
    positive_lift_rate = float((successful["ranker_ndcg_lift_vs_hybrid"] > 0).mean()) if len(successful) else 0.0
    add("positive_ranker_lift_rate", positive_lift_rate >= acceptance["positive_ranker_lift_rate_min"], positive_lift_rate, acceptance["positive_ranker_lift_rate_min"], "The learned ranker should usually improve hybrid retrieval.")
    median_lift = float(successful["ranker_ndcg_lift_vs_hybrid"].median()) if len(successful) else float("nan")
    add("median_ranker_ndcg_lift", np.isfinite(median_lift) and median_lift >= acceptance["median_ranker_ndcg_lift_min"], median_lift, acceptance["median_ranker_ndcg_lift_min"], "Median lift guards against one lucky seed.")
    ndcg_p05 = _quantile(successful["ndcg_at_10"], 0.05)
    add("ndcg_at_10_lower_tail", np.isfinite(ndcg_p05) and ndcg_p05 >= acceptance["ndcg_at_10_p05_min"], ndcg_p05, acceptance["ndcg_at_10_p05_min"], "The lower tail should remain useful under stress scenarios.")
    efficiency_p05 = _quantile(successful["recall_efficiency_at_10"], 0.05)
    add("recall_efficiency_lower_tail", np.isfinite(efficiency_p05) and efficiency_p05 >= acceptance["recall_efficiency_at_10_p05_min"], efficiency_p05, acceptance["recall_efficiency_at_10_p05_min"], "Top-k should recover a useful fraction of the maximum relevant items that can fit in top-k.")
    ece_p95 = _quantile(successful["ece"], 0.95)
    add("ece_upper_tail", np.isfinite(ece_p95) and ece_p95 <= acceptance["ece_p95_max"], ece_p95, acceptance["ece_p95_max"], "Calibration error should remain controlled.")
    latency_p95 = _quantile(successful["p95_latency_ms"], 0.95)
    add("latency_upper_tail", np.isfinite(latency_p95) and latency_p95 <= acceptance["p95_latency_p95_max_ms"], latency_p95, acceptance["p95_latency_p95_max_ms"], "Small-data serving latency must stay within the configured local budget.")
    self_return_p95 = _quantile(successful["similar_to_self_return_at_10"], 0.95)
    add("similar_to_self_return_upper_tail", np.isfinite(self_return_p95) and self_return_p95 <= acceptance["similar_to_self_return_p95_max"], self_return_p95, acceptance["similar_to_self_return_p95_max"], "A similar-to query should not simply return its anchor item.")
    blocks = int((successful["launch_decision"] == "BLOCK").sum()) if len(successful) else 0
    add("no_block_decisions", (blocks == 0) if acceptance.get("no_block_decisions", True) else True, blocks, 0, "No approved-data Monte Carlo trial should violate a hard claim/privacy boundary.")

    split_integrity_rate = float((successful["split_anchor_leakage_free"] & successful["split_personalized_user_leakage_free"]).mean()) if len(successful) else 0.0
    add("split_integrity_rate", split_integrity_rate >= acceptance.get("split_integrity_rate_min", 1.0), split_integrity_rate, acceptance.get("split_integrity_rate_min", 1.0), "Anchor and personalized-user groups should not cross data splits.")
    clean_truth_rate = float((successful["evaluation_truth"] == "clean_label").mean()) if len(successful) else 0.0
    add("clean_truth_evaluation_rate", clean_truth_rate >= acceptance.get("clean_truth_evaluation_rate_min", 1.0), clean_truth_rate, acceptance.get("clean_truth_evaluation_rate_min", 1.0), "Synthetic trials should evaluate against latent clean relevance rather than noisy observed labels.")

    # Directional scenario checks use paired common-random-number trials rather
    # than comparing unrelated scenario means. This isolates the stress effect.
    long_tail_delta = _paired_deltas(successful, "nominal", "long_tail_skew", "item_popularity_gini")
    if len(long_tail_delta):
        median_delta = float(long_tail_delta.median())
        add("long_tail_scenario_is_more_skewed", median_delta > 0, median_delta, "> 0", "Paired long-tail trials should increase popularity inequality.")

    sparse_coverage_delta = _paired_deltas(successful, "nominal", "sparse_metadata", "tag_coverage")
    if len(sparse_coverage_delta):
        median_delta = float(sparse_coverage_delta.median())
        add("sparse_metadata_has_lower_coverage", median_delta < 0, median_delta, "< 0", "Paired sparse-metadata trials should reduce observed tag coverage.")
    sparse_visual_delta = _paired_deltas(successful, "nominal", "sparse_metadata", "visual_query_hybrid_ndcg_all")
    if len(sparse_visual_delta):
        median_delta = float(sparse_visual_delta.median())
        add("sparse_metadata_reduces_visual_query_quality", median_delta < 0, median_delta, "< 0", "Paired sparse-metadata trials should reduce visual-query quality.")

    noise_rating_delta = _paired_deltas(successful, "nominal", "high_noise", "rating_std")
    if len(noise_rating_delta):
        median_delta = float(noise_rating_delta.median())
        add("high_noise_has_higher_rating_variance", median_delta > 0, median_delta, "> 0", "Paired high-noise trials should increase rating dispersion.")
    noise_corr_delta = _paired_deltas(successful, "nominal", "high_noise", "preference_rating_correlation")
    if len(noise_corr_delta):
        median_delta = float(noise_corr_delta.median())
        add("high_noise_weakens_preference_signal", median_delta < 0, median_delta, "< 0", "Paired high-noise trials should weaken preference-to-rating correlation.")
    noise_ndcg_delta = _paired_deltas(successful, "nominal", "high_noise", "ndcg_at_10")
    if len(noise_ndcg_delta):
        median_delta = float(noise_ndcg_delta.median())
        degradation_rate = float((noise_ndcg_delta < 0).mean())
        add("high_noise_reduces_ranking_quality_paired_median", median_delta < 0, median_delta, "< 0", "Paired high-noise trials should reduce median clean-truth NDCG.")
        min_rate = float(acceptance.get("high_noise_degradation_rate_min", 0.5))
        add("high_noise_degradation_rate", degradation_rate >= min_rate, degradation_rate, min_rate, "A majority of paired high-noise trials should degrade ranking quality.")
    noise_disagreement_delta = _paired_deltas(successful, "nominal", "high_noise", "label_disagreement_rate")
    if len(noise_disagreement_delta):
        median_delta = float(noise_disagreement_delta.median())
        add("high_noise_increases_label_disagreement", median_delta > 0, median_delta, "> 0", "Paired high-noise trials should increase observed-vs-clean label disagreement.")
    return checks


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return df.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo end-to-end validation")
    parser.add_argument("--config", default="configs/monte_carlo.yaml")
    parser.add_argument("--base-config", default="configs/pipeline.yaml")
    parser.add_argument("--trials-per-scenario", type=int, default=None)
    parser.add_argument("--keep-trials", action="store_true")
    args = parser.parse_args()

    mc_cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load((ROOT / args.base_config).read_text(encoding="utf-8"))
    trials_per_scenario = args.trials_per_scenario or int(mc_cfg["trials_per_scenario"])
    keep_trials = args.keep_trials or bool(mc_cfg.get("keep_trial_artifacts", False))
    scenarios = mc_cfg["scenarios"]
    # Common random numbers: the same trial seed is reused across scenarios.
    # This pairs nominal/stress runs on comparable latent catalogs and user draws,
    # reducing scenario-comparison variance from unrelated random datasets.
    seed_rng = random.Random(int(mc_cfg["base_seed"]))
    trial_seeds = [seed_rng.randrange(1, 2**32 - 1) for _ in range(trials_per_scenario)]

    work_root = ROOT / "artifacts" / "monte_carlo_work"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario_name, scenario in scenarios.items():
        for trial_index in range(trials_per_scenario):
            seed = trial_seeds[trial_index]
            trial_dir = work_root / f"{scenario_name}_{trial_index:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            cfg = copy.deepcopy(base_cfg)
            cfg["project"]["random_seed"] = seed
            sampled_scenario = sample_scenario_parameters(
                scenario, seed + 99173, mc_cfg.get("parameter_sampling", {})
            )
            small = mc_cfg["small_data"]
            cfg["data"].update({
                "demo_movies": int(small["demo_movies"]),
                "demo_users": int(small["demo_users"]),
                "demo_ratings": int(small["demo_ratings"]),
            })
            cfg["data"]["synthetic"].update({k: v for k, v in sampled_scenario.items() if k != "label_noise"})
            cfg["queries"].update({
                "num_queries": int(small["num_queries"]),
                "candidates_per_query": int(small["candidates_per_query"]),
                "label_all_catalog_demo": True,
                "label_noise": float(sampled_scenario.get("label_noise", 0.0)),
            })
            cfg["retrieval"].update({"top_k": int(small["retrieval_top_k"]), "dense_backend": "tfidf", "faiss_use_gpu_if_available": False})
            cfg["ranking"].update({"n_estimators": int(small["ranker_estimators"]), "device_type": "cpu", "n_jobs": 1})
            cfg.setdefault("evaluation", {})["latency_max_queries"] = 2
            cfg_path = trial_dir / "pipeline.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            start = time.perf_counter()
            old_cwd = Path.cwd()
            record = {
                "scenario": scenario_name, "trial": trial_index, "seed": seed, "success": False, "error": "",
                **{f"param_{key}": value for key, value in sampled_scenario.items()},
            }
            try:
                os.chdir(trial_dir)
                result = run_pipeline(cfg_path, mode="demo")
                metrics = result["metrics"]
                diagnostics = result["data_diagnostics"]
                query_type_metrics = result.get("query_type_metrics", {})
                diagnostic_metrics = result.get("diagnostic_metrics", {})
                record.update({
                    "success": True,
                    "launch_decision": result["launch_decision"],
                    "ranker_backend": result["ranker_backend"],
                    "calibration_method": result["calibration_method"],
                    "ranker_ndcg_lift_vs_hybrid": result["ranker_ndcg_lift_vs_hybrid"],
                    "similar_to_self_return_at_10": result.get("similar_to_self_return_at_10", 0.0),
                    "visual_query_ndcg_at_10": query_type_metrics.get("visual_query", {}).get("ndcg_at_10", float("nan")),
                    "visual_query_hybrid_ndcg_all": diagnostic_metrics.get("visual_query_hybrid_ndcg_all", float("nan")),
                    "mood_decade_ndcg_at_10": query_type_metrics.get("mood_decade", {}).get("ndcg_at_10", float("nan")),
                    "evaluation_truth": result.get("evaluation_truth"),
                    "split_anchor_leakage_free": bool(result.get("split_diagnostics", {}).get("anchor_leakage_free", False)),
                    "split_personalized_user_leakage_free": bool(result.get("split_diagnostics", {}).get("personalized_user_leakage_free", False)),
                    **metrics,
                    **diagnostics,
                })
                record["metric_bounds_ok"] = _metric_bounds_ok(record)
            except Exception as exc:
                record.update({"success": False, "launch_decision": "ERROR", "metric_bounds_ok": False, "error": f"{type(exc).__name__}: {exc}"})
                (trial_dir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                os.chdir(old_cwd)
                record["runtime_seconds"] = time.perf_counter() - start
                records.append(record)
                if "result" in locals():
                    del result
            print(f"[{scenario_name} {trial_index + 1}/{trials_per_scenario}] success={record['success']} decision={record.get('launch_decision')} ndcg={record.get('ndcg_at_10')}", flush=True)

    trials = pd.DataFrame(records)
    summary = _scenario_summary(trials[trials["success"]])
    checks = pd.DataFrame(_checks(trials, summary, mc_cfg["acceptance"]))
    overall = "PASS" if len(checks) and checks["passed"].all() else "REVIEW"

    artifacts_dir = ROOT / "artifacts"
    reports_dir = ROOT / "reports"
    artifacts_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    trials.to_csv(artifacts_dir / "monte_carlo_trials.csv", index=False)
    summary.to_csv(artifacts_dir / "monte_carlo_scenario_summary.csv", index=False)
    checks.to_csv(artifacts_dir / "monte_carlo_checks.csv", index=False)
    payload = {
        "decision": overall,
        "total_trials": int(len(trials)),
        "successful_trials": int(trials["success"].sum()),
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": int(len(checks)),
    }
    (artifacts_dir / "monte_carlo_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failures = checks[~checks["passed"]] if len(checks) else checks
    report = [
        "# Monte Carlo End-to-End Validation Report", "",
        f"**Decision:** {overall}", "",
        f"Independent trials: {len(trials)} across {len(scenarios)} scenarios.", "",
        "## Scenarios", "",
        _markdown_table(pd.DataFrame([{"scenario": name, **values} for name, values in scenarios.items()])), "",
        "## Parameter uncertainty", "",
        f"Sampling configuration: `{json.dumps(mc_cfg.get('parameter_sampling', {}), sort_keys=True)}`", "",
        "## Acceptance checks", "", _markdown_table(checks), "",
        "## Scenario-level metric distribution", "", _markdown_table(summary), "",
        "## Trial-level results", "", _markdown_table(trials.drop(columns=["error"], errors="ignore")), "",
        "## Analysis", "",
        "The framework is considered reasonable only when all trials execute end to end, metrics remain finite and bounded, the learned ranker usually improves hybrid retrieval, calibration and latency remain controlled, and scenario parameters change the synthetic data in the intended direction.", "",
        "## Methodology and limits", "",
        f"This is a lightweight Monte Carlo stress audit over synthetic catalogs, preferences, interactions, query intents, metadata sparsity, popularity skew, and label/rating noise. This run used {trials_per_scenario} paired trial(s) per scenario. Scenario percentiles are descriptive at small trial counts. Increase `--trials-per-scenario` for stronger uncertainty estimates. Synthetic results do not replace MovieLens or human-judged search evaluation.", "",
    ]
    if len(failures):
        report.extend(["## Items requiring review", "", _markdown_table(failures), ""])
    else:
        report.extend(["## Items requiring review", "", "No material Monte Carlo acceptance failure was detected.", ""])
    (reports_dir / "14_monte_carlo_validation_report.md").write_text("\n".join(report), encoding="utf-8")

    if not keep_trials:
        shutil.rmtree(work_root, ignore_errors=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
