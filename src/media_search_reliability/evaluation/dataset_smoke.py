from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from media_search_reliability.data_ingestion.synthetic_data import generate_synthetic_movielens
from media_search_reliability.pipeline import run_pipeline

SUPPORTED_FORMATS = ("dat", "csv")


def _write_dat_fixture(raw_dir: Path, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Write a MovieLens 1M-compatible fixture without tags.dat.

    Official MovieLens 1M uses movies.dat, ratings.dat, and users.dat. The
    absence of tags.dat is intentional and validates the framework's no-tag path.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "movies.dat").open("w", encoding="latin-1", errors="replace") as handle:
        for row in movies.sort_values("movie_id").itertuples(index=False):
            handle.write(f"{int(row.movie_id)}::{row.title}::{row.genres}\n")
    with (raw_dir / "ratings.dat").open("w", encoding="latin-1", errors="replace") as handle:
        ordered = ratings.sort_values(["user_id", "movie_id"], kind="mergesort")
        for index, row in enumerate(ordered.itertuples(index=False)):
            rating = int(min(5, max(1, round(float(row.rating)))))
            handle.write(
                f"{int(row.user_id)}::{int(row.movie_id)}::{rating}::{978300000 + index}\n"
            )
    with (raw_dir / "users.dat").open("w", encoding="latin-1") as handle:
        for user_id in sorted(int(value) for value in ratings["user_id"].unique()):
            handle.write(f"{user_id}::M::25::12::00000\n")


def _write_csv_fixture(
    raw_dir: Path,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
) -> None:
    """Write a modern MovieLens CSV-compatible fixture with tags."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    movies[["movie_id", "title", "genres"]].rename(
        columns={"movie_id": "movieId"}
    ).to_csv(raw_dir / "movies.csv", index=False)
    ratings[["user_id", "movie_id", "rating"]].rename(
        columns={"user_id": "userId", "movie_id": "movieId"}
    ).assign(timestamp=978300000).to_csv(raw_dir / "ratings.csv", index=False)
    tags[["user_id", "movie_id", "tag"]].rename(
        columns={"user_id": "userId", "movie_id": "movieId"}
    ).assign(timestamp=978300000).to_csv(raw_dir / "tags.csv", index=False)


def _load_packaged_config() -> dict:
    resource = resources.files("media_search_reliability.configs").joinpath("pipeline.yaml")
    with resources.as_file(resource) as config_path:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _build_config(run_root: Path, *, quick: bool) -> Path:
    config = _load_packaged_config()
    config["project"]["data_dir"] = str((run_root / "data").resolve())
    config["project"]["output_dir"] = str((run_root / "artifacts").resolve())
    config["project"]["reports_dir"] = str((run_root / "reports").resolve())
    config["data"]["max_movies"] = 180 if quick else 260
    config["data"]["max_users"] = 90 if quick else 140
    config["queries"]["num_queries"] = 30 if quick else 50
    config["queries"]["candidates_per_query"] = 20 if quick else 28
    config["retrieval"]["top_k"] = 20 if quick else 28
    config["ranking"]["n_estimators"] = 30 if quick else 45
    config["ranking"]["min_child_samples"] = 8
    config["evaluation"]["latency_max_queries"] = 6 if quick else 10
    config["evaluation"]["latency_warmup_queries"] = 1
    config_path = run_root / "pipeline_dataset_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _validate_run(run_root: Path, summary: dict, fixture_format: str) -> dict:
    required = [
        run_root / "artifacts" / "eval_summary.json",
        run_root / "artifacts" / "launch_gate.json",
        run_root / "artifacts" / "ranker_bundle.joblib",
        run_root / "artifacts" / "retrieval_bundle.joblib",
        run_root / "reports" / "01_data_validation_report.md",
        run_root / "reports" / "03_lambdarank_training_report.md",
        run_root / "reports" / "07_launch_readiness_memo.md",
        run_root / "reports" / "09_claim_boundary.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    ranker_backend = str(summary.get("ranker_backend", ""))
    split = summary.get("split_diagnostics", {})
    decision = str(summary.get("launch_decision", ""))
    checks = {
        "pipeline_completed": bool(summary),
        "artifact_contract": not missing,
        "ranker_backend_contract": ranker_backend.startswith("lightgbm-lambdarank"),
        "anchor_leakage_free": bool(split.get("anchor_leakage_free", False)),
        "personalized_user_leakage_free": bool(
            split.get("personalized_user_leakage_free", False)
        ),
        "valid_launch_decision": decision in {"PASS", "REVIEW", "ITERATE", "BLOCK"},
        "public_data_evaluation_truth": summary.get("evaluation_truth") == "label",
    }
    passed = all(checks.values())
    return {
        "format": fixture_format,
        "passed": passed,
        "decision": decision,
        "ranker_backend": ranker_backend,
        "dense_backend": summary.get("dense_backend"),
        "vector_index_backend": summary.get("vector_index_backend"),
        "ndcg_at_10": float(summary.get("metrics", {}).get("ndcg_at_10", 0.0)),
        "ranker_ndcg_lift_vs_hybrid": float(
            summary.get("metrics", {}).get("ranker_ndcg_lift_vs_hybrid", 0.0)
        ),
        "missing_artifacts": missing,
        "checks": checks,
    }


def run_dataset_smoke(
    *,
    formats: Iterable[str] = SUPPORTED_FORMATS,
    quick: bool = False,
    seed: int = 20260612,
    output_root: str | Path | None = None,
    keep_work: bool = False,
) -> dict:
    selected = tuple(dict.fromkeys(str(value).strip().lower() for value in formats))
    invalid = sorted(set(selected) - set(SUPPORTED_FORMATS))
    if invalid:
        raise ValueError(f"Unsupported fixture formats: {invalid}")
    if not selected:
        raise ValueError("At least one fixture format is required.")

    cwd = Path.cwd()
    output_root_path = Path(output_root).resolve() if output_root else cwd
    artifact_dir = output_root_path / "artifacts"
    report_dir = output_root_path / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if keep_work:
        work_root = output_root_path / "dataset_smoke_work"
        shutil.rmtree(work_root, ignore_errors=True)
        work_root.mkdir(parents=True, exist_ok=True)
        temporary_context = None
    else:
        temporary_context = tempfile.TemporaryDirectory(prefix="media-search-dataset-smoke-")
        work_root = Path(temporary_context.name)

    n_movies = 180 if quick else 260
    n_users = 90 if quick else 140
    n_ratings = 2600 if quick else 4800
    movies, ratings, tags = generate_synthetic_movielens(
        n_movies=n_movies,
        n_users=n_users,
        n_ratings=n_ratings,
        seed=seed,
        popularity_alpha=1.05,
        preference_strength=1.35,
        rating_noise=0.70,
        tag_observation_prob=0.60,
        exploration_rate=0.12,
        cold_start_fraction=0.08,
    )

    results: list[dict] = []
    try:
        for offset, fixture_format in enumerate(selected):
            run_root = work_root / fixture_format
            raw_dir = run_root / "data" / "raw" / "movielens"
            if fixture_format == "dat":
                _write_dat_fixture(raw_dir, movies, ratings)
            else:
                _write_csv_fixture(raw_dir, movies, ratings, tags)
            config_path = _build_config(run_root, quick=quick)
            summary = run_pipeline(config_path, mode="movielens")
            result = _validate_run(run_root, summary, fixture_format)
            results.append(result)
            print(
                f"[{fixture_format}] passed={result['passed']} "
                f"decision={result['decision']} ndcg@10={result['ndcg_at_10']:.4f} "
                f"ranker_lift={result['ranker_ndcg_lift_vs_hybrid']:+.4f}",
                flush=True,
            )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()

    payload = {
        "decision": "PASS" if all(item["passed"] for item in results) else "FAIL",
        "quick": bool(quick),
        "formats": list(selected),
        "results": results,
        "interpretation": (
            "PASS means both MovieLens file-layout paths completed ingestion, ranking, "
            "evaluation, and artifact generation. A launch decision of REVIEW or ITERATE "
            "is a data-quality/model-quality outcome, not a runnability failure."
        ),
    }
    (artifact_dir / "dataset_smoke_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Dataset-Only Local and GitHub Runnability Verification",
        "",
        "This audit validates the public-data execution path using two small, faithful MovieLens layouts:",
        "",
        "- `dat`: MovieLens 1M-style `movies.dat`, `ratings.dat`, and `users.dat`, intentionally without tags.",
        "- `csv`: modern MovieLens-style `movies.csv`, `ratings.csv`, and `tags.csv`.",
        "",
        f"Overall decision: **{payload['decision']}**",
        "",
        "| Format | Audit | Launch output | Ranker backend | NDCG@10 | Ranker lift vs hybrid |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['format']} | {'PASS' if item['passed'] else 'FAIL'} | "
            f"{item['decision']} | {item['ranker_backend']} | "
            f"{item['ndcg_at_10']:.4f} | {item['ranker_ndcg_lift_vs_hybrid']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "The fixture data is generated only to exercise the exact public-data loaders and end-to-end path. "
            "It is not a substitute for final MovieLens benchmark metrics.",
        ]
    )
    (report_dir / "21_dataset_only_runnability_verification.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-independent MovieLens dat/csv end-to-end dataset smoke tests."
    )
    parser.add_argument(
        "--formats",
        default="dat,csv",
        help="Comma-separated subset of: dat,csv",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    payload = run_dataset_smoke(
        formats=[value for value in args.formats.split(",") if value.strip()],
        quick=args.quick,
        seed=args.seed,
        output_root=args.output_root,
        keep_work=args.keep_work,
    )
    print(json.dumps(payload, indent=2))
    if payload["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
