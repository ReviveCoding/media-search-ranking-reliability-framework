from __future__ import annotations

import argparse
import compileall
import json
from pathlib import Path

import media_search_reliability
from media_search_reliability.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REPOSITORY_FILES = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "Dockerfile",
    "Makefile",
    ".github/workflows/ci.yml",
    "configs/pipeline.yaml",
    "configs/pipeline_gpu.yaml",
    "configs/monte_carlo.yaml",
    "scripts/api_smoke_test.py",
    "scripts/reproducibility_check.py",
    "scripts/monte_carlo_validate.py",
]
REQUIRED_ARTIFACTS = [
    "artifacts/eval_summary.json",
    "artifacts/ranker_bundle.joblib",
    "artifacts/retrieval_bundle.joblib",
    "artifacts/test_predictions.csv",
    "reports/07_launch_readiness_memo.md",
    "reports/08_model_card.md",
    "reports/09_claim_boundary.md",
    "reports/16_split_integrity_report.md",
    "reports/17_evaluation_truth_report.md",
    "reports/18_reproducibility_report.md",
    "reports/19_final_weakness_resolution_and_optimization.md",
    "reports/20_release_hardening_and_final_audit.md",
]


def _status_row(check: str, passed: bool, detail: str) -> dict:
    return {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one in-process end-to-end audit plus repository contract checks.")
    parser.add_argument("--mode", default="demo", choices=["demo", "movielens"])
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    rows: list[dict] = []
    rows.append(_status_row("installed_package_import", True, f"media_search_reliability {media_search_reliability.__version__}"))

    compile_ok = compileall.compile_dir(ROOT / "src", quiet=1) and compileall.compile_dir(ROOT / "scripts", quiet=1)
    rows.append(_status_row("source_compilation", bool(compile_ok), "src/ and scripts/ compile"))

    missing_repo = [path for path in REQUIRED_REPOSITORY_FILES if not (ROOT / path).exists()]
    rows.append(_status_row("repository_contract", not missing_repo, "missing=" + json.dumps(missing_repo)))

    summary = run_pipeline(ROOT / args.config, mode=args.mode)
    pipeline_ok = summary.get("launch_decision") in {"PASS", "REVIEW"}
    rows.append(_status_row("end_to_end_pipeline", pipeline_ok, f"launch={summary.get('launch_decision')}"))

    split = summary.get("split_diagnostics", {})
    leakage_ok = bool(split.get("anchor_leakage_free")) and bool(split.get("personalized_user_leakage_free"))
    rows.append(_status_row("split_integrity", leakage_ok, json.dumps({
        "anchor_leakage_free": split.get("anchor_leakage_free"),
        "personalized_user_leakage_free": split.get("personalized_user_leakage_free"),
    })))

    missing_outputs = [path for path in REQUIRED_ARTIFACTS if not (ROOT / path).exists()]
    rows.append(_status_row("artifact_contract", not missing_outputs, "missing=" + json.dumps(missing_outputs)))

    checks_pass = all(row["status"] == "PASS" for row in rows)
    lines = [
        "# Local and GitHub Runnability Audit",
        "",
        f"**Decision:** {'PASS' if checks_pass else 'FAIL'}",
        "",
        "This audit intentionally runs one end-to-end pipeline in-process and checks the repository/artifact contract. Unit tests, API smoke, reproducibility replay, package build, and Monte Carlo validation run as separate processes in CI to avoid OpenMP/TestClient teardown interference in constrained shells.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in rows:
        detail = str(row["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['check']} | {row['status']} | {detail} |")
    lines.extend([
        "",
        "## Separate full-validation commands",
        "",
        "```bash",
        "python -m pytest -q",
        "python scripts/local_audit.py --mode demo",
        "python scripts/api_smoke_test.py",
        "python scripts/reproducibility_check.py",
        "python scripts/monte_carlo_validate.py --trials-per-scenario 1",
        "python -m build",
        "```",
        "",
    ])
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "13_local_github_runnability_audit.md").write_text("\n".join(lines), encoding="utf-8")

    for row in rows:
        print(f"{row['status']:4} {row['check']}: {row['detail']}")
    if not checks_pass:
        raise SystemExit("Local audit failed")
    print("Local audit PASS")


if __name__ == "__main__":
    main()
