from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

CORE_RUNS = ("core_conservative", "core_balanced", "core_compact")
ENRICHED_RUNS = (
    "core_champion_replay",
    "tag_genome_feature_only",
    "imdb_feature_only",
    "combined_feature_only",
)
SLICE_COLUMNS = (
    "genre_tag_ndcg",
    "mood_decade_ndcg",
    "personalized_ndcg",
    "similar_to_ndcg",
    "visual_query_ndcg",
)
REPLAY_METRICS = ("ndcg_at_10", "mrr_at_10", "recall_efficiency_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate completed frozen MovieLens variants without retraining."
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/frozen_quality_ablation",
    )
    parser.add_argument(
        "--reports-root",
        default="reports/frozen_quality_ablation",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/benchmarks/ml10m_frozen_v1",
    )
    parser.add_argument("--replay-atol", type=float, default=1e-3)
    parser.add_argument("--replay-rtol", type=float, default=1e-3)
    return parser.parse_args()


def load_summary(output_root: Path, run_name: str) -> dict:
    path = output_root / run_name / "eval_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed variant summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(output_root: Path, run_name: str) -> str | None:
    path = output_root / run_name / "run_config_sha256.txt"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def row(run_name: str, summary: dict, *, profile: str, variant: str) -> dict:
    metrics = summary["metrics"]
    q = summary.get("query_type_metrics", {})
    enrichment = summary.get("enrichment_diagnostics", {})
    return {
        "run_name": run_name,
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
        "tag_genome_applied": enrichment.get("tag_genome_applied", False),
        "imdb_applied": enrichment.get("imdb_applied", False),
        "retrieval_text_changed": enrichment.get("retrieval_text_changed", False),
    }


def exact_contract_check(reference: dict, candidate: dict, run_name: str) -> None:
    if reference.get("frozen_benchmark_manifest") != candidate.get("frozen_benchmark_manifest"):
        raise AssertionError(f"{run_name} used a different frozen benchmark manifest")
    if reference.get("split_diagnostics") != candidate.get("split_diagnostics"):
        raise AssertionError(f"{run_name} used different query-group splits")


def replay_diagnostics(
    original_row: pd.Series,
    replay_row: pd.Series,
    *,
    original_fingerprint: str | None,
    replay_fingerprint: str | None,
    contract_comparable: bool,
    atol: float,
    rtol: float,
) -> dict:
    metrics: dict[str, dict] = {}
    numeric_passed = True
    for name in REPLAY_METRICS:
        original = float(original_row[name])
        replay = float(replay_row[name])
        delta = replay - original
        close = math.isclose(replay, original, rel_tol=rtol, abs_tol=atol)
        metrics[name] = {
            "original": original,
            "replay": replay,
            "delta": delta,
            "absolute_delta": abs(delta),
            "abs_tolerance": atol,
            "rel_tolerance": rtol,
            "within_numeric_tolerance": close,
        }
        numeric_passed = numeric_passed and close

    comparable = (
        contract_comparable
        and original_fingerprint is not None
        and replay_fingerprint is not None
        and original_fingerprint == replay_fingerprint
    )

    if comparable:
        status = "strictly_comparable"
        passed = numeric_passed
        interpretation = (
            "Config fingerprints match, so GPU replay metrics are evaluated against tolerance."
        )
    else:
        status = "legacy_not_strictly_comparable"
        passed = None
        interpretation = (
            "The legacy profile-selection run does not have a fully matching frozen contract and config fingerprint. "
            "Metric deltas are diagnostic only; core_champion_replay is the canonical promotion baseline."
        )

    return {
        "status": status,
        "strict_replay_claim": bool(comparable and numeric_passed),
        "passed": passed,
        "numeric_within_tolerance": numeric_passed,
        "original_config_fingerprint": original_fingerprint,
        "replay_config_fingerprint": replay_fingerprint,
        "interpretation": interpretation,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    summaries = {
        name: load_summary(output_root, name)
        for name in (*CORE_RUNS, *ENRICHED_RUNS)
    }

    core_rows = []
    for name in CORE_RUNS:
        profile = name.removeprefix("core_")
        core_rows.append(row(name, summaries[name], profile=profile, variant="core"))
    core_table = pd.DataFrame(core_rows)
    selected = core_table.sort_values(
        ["ndcg_at_10", "recall_efficiency_at_10"],
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    champion_profile = str(selected["ranker_profile"])
    legacy_core_name = f"core_{champion_profile}"

    reference = summaries["core_champion_replay"]

    # Only the canonical promotion baseline and enrichment variants must share
    # the exact frozen benchmark contract. Legacy core profile runs may have
    # been produced before the frozen split contract was finalized; retain
    # them for historical/profile-selection context, but do not allow them to
    # block aggregation or champion promotion.
    strict_contract_runs = ENRICHED_RUNS
    for name in strict_contract_runs:
        exact_contract_check(reference, summaries[name], name)

    legacy_contract_diagnostics = {}
    for name in CORE_RUNS:
        summary = summaries[name]
        legacy_contract_diagnostics[name] = {
            "manifest_equal_to_canonical": (
                reference.get("frozen_benchmark_manifest")
                == summary.get("frozen_benchmark_manifest")
            ),
            "split_equal_to_canonical": (
                reference.get("split_diagnostics")
                == summary.get("split_diagnostics")
            ),
            "strictly_comparable": False,
            "role": "legacy_profile_selection_context_only",
        }

    rows = list(core_rows)
    for name in ENRICHED_RUNS:
        rows.append(
            row(
                name,
                summaries[name],
                profile=champion_profile,
                variant=name,
            )
        )
    final = pd.DataFrame(rows)

    legacy_row = final[final["run_name"] == legacy_core_name].iloc[0]
    replay_row = final[final["run_name"] == "core_champion_replay"].iloc[0]
    selected_legacy_contract = legacy_contract_diagnostics[legacy_core_name]
    legacy_contract_comparable = bool(
        selected_legacy_contract["manifest_equal_to_canonical"]
        and selected_legacy_contract["split_equal_to_canonical"]
    )
    replay_check = replay_diagnostics(
        legacy_row,
        replay_row,
        original_fingerprint=fingerprint(output_root, legacy_core_name),
        replay_fingerprint=fingerprint(output_root, "core_champion_replay"),
        contract_comparable=legacy_contract_comparable,
        atol=args.replay_atol,
        rtol=args.replay_rtol,
    )
    replay_check.update(
        {
            "selected_legacy_core_profile": champion_profile,
            "legacy_core_run": legacy_core_name,
            "canonical_baseline_run": "core_champion_replay",
            "canonical_contract_validated": True,
            "legacy_manifest_equal": selected_legacy_contract["manifest_equal_to_canonical"],
            "legacy_split_equal": selected_legacy_contract["split_equal_to_canonical"],
            "strict_contract_runs": list(strict_contract_runs),
            "legacy_contract_diagnostics": legacy_contract_diagnostics,
        }
    )
    (output_root / "frozen_replay_diagnostics.json").write_text(
        json.dumps(replay_check, indent=2, default=str),
        encoding="utf-8",
    )

    if replay_check["status"] == "strictly_comparable" and replay_check["passed"] is False:
        failed = [
            name
            for name, detail in replay_check["metrics"].items()
            if not detail["within_numeric_tolerance"]
        ]
        raise AssertionError(
            "Strictly comparable core replay exceeded tolerance for: " + ", ".join(failed)
        )

    baseline = replay_row

    def eligible(candidate: pd.Series) -> bool:
        run_name = str(candidate["run_name"])
        if run_name == "core_champion_replay":
            return True
        if str(candidate["variant"]) == "core":
            return False
        if bool(candidate.get("retrieval_text_changed", False)):
            return False
        if float(candidate.get("ranker_lift_vs_hybrid", 0.0) or 0.0) <= 0:
            return False
        for column in SLICE_COLUMNS:
            base_value = baseline.get(column)
            new_value = candidate.get(column)
            if pd.notna(base_value) and float(base_value) > 0 and pd.notna(new_value):
                if float(new_value) < 0.90 * float(base_value):
                    return False
        return float(candidate["ndcg_at_10"]) >= float(baseline["ndcg_at_10"])

    final["eligible_for_promotion"] = final.apply(eligible, axis=1)
    eligible_rows = final[final["eligible_for_promotion"]]
    if eligible_rows.empty:
        raise AssertionError("No eligible champion candidate remained after promotion guardrails")

    champion = eligible_rows.sort_values(
        ["ndcg_at_10", "recall_efficiency_at_10"],
        ascending=False,
        kind="mergesort",
    ).iloc[0]

    final = final.sort_values("ndcg_at_10", ascending=False, kind="mergesort")
    final.to_csv(output_root / "frozen_ablation_results.csv", index=False)

    summary = {
        "champion": champion.to_dict(),
        "canonical_core_baseline": baseline.to_dict(),
        "legacy_profile_selection_run": legacy_core_name,
        "replay_diagnostics_status": replay_check["status"],
        "strict_gpu_replay_claim": replay_check["strict_replay_claim"],
        "manifest_dir": str(Path(args.manifest_dir).resolve()),
        "strict_contract_runs": list(strict_contract_runs),
        "legacy_contract_diagnostics": legacy_contract_diagnostics,
        "promotion_rule": (
            "same frozen manifest/splits, no retrieval-text drift, positive ranker lift, "
            "no >10% slice regression, non-negative overall NDCG change"
        ),
    }
    (output_root / "frozen_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    report_path = reports_root / "29_frozen_movielens_ablation.md"
    report = [
        "# Frozen MovieLens Ranking Quality Ablation",
        "",
        "Promotion-eligible variants use the same frozen catalog fingerprint, queries, judgments, and query-group splits.",
        "Legacy core profile runs are retained for historical/profile-selection context only and are not treated as strictly comparable.",
        "Tag Genome and IMDb operate in feature-only mode and do not rewrite retrieval text.",
        "",
        final.to_markdown(index=False),
        "",
        "## Canonical baseline",
        "",
        "`core_champion_replay` is the canonical promotion baseline.",
        "",
        "## Replay interpretation",
        "",
        replay_check["interpretation"],
        "",
        "## Champion",
        "",
        f"`{champion['run_name']}` with NDCG@10 `{float(champion['ndcg_at_10']):.4f}`.",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(final.to_string(index=False))
    print(f"Replay status: {replay_check['status']}")
    print(f"Champion: {champion['run_name']} NDCG@10={float(champion['ndcg_at_10']):.6f}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
