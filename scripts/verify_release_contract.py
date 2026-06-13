from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


README_START = "<!-- FINAL_RESULTS_START -->"
README_END = "<!-- FINAL_RESULTS_END -->"

EXPECTED_DOCS = [
    "docs/FINAL_RESULTS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/CLAIM_BOUNDARIES.md",
]

EXPECTED_STRICT_RUNS = {
    "core_champion_replay",
    "tag_genome_feature_only",
    "imdb_feature_only",
    "combined_feature_only",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    return value


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def git_tracked_large_files(repo: Path) -> list[tuple[str, int]]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []

    large: list[tuple[str, int]] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="replace")
        path = repo / relative
        if path.is_file():
            size = path.stat().st_size
            if size >= 100 * 1024 * 1024:
                large.append((relative, size))
    return large


def verify(repo: Path) -> dict:
    artifact_dir = repo / "artifacts" / "frozen_quality_ablation"
    summary = load_json(artifact_dir / "frozen_ablation_summary.json")
    diagnostics = load_json(artifact_dir / "frozen_replay_diagnostics.json")
    finalization = load_json(
        repo / "reports" / "repository_finalization" / "finalization_summary.json"
    )

    champion = summary["champion"]
    baseline = summary["canonical_core_baseline"]

    assert champion["run_name"] == "combined_feature_only"
    assert baseline["run_name"] == "core_champion_replay"
    assert champion["eligible_for_promotion"] is True
    assert diagnostics["canonical_contract_validated"] is True
    assert diagnostics["strict_replay_claim"] is False
    assert diagnostics["status"] == "legacy_not_strictly_comparable"
    assert set(summary["strict_contract_runs"]) == EXPECTED_STRICT_RUNS
    assert finalization["status"] == "PASS"

    readme_path = repo / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(readme_path)
    readme = readme_path.read_text(encoding="utf-8")
    assert README_START in readme and README_END in readme
    assert "`combined_feature_only`" in readme
    assert f"`{float(champion['ndcg_at_10']):.6f}`" in readme
    assert "`core_champion_replay`" in readme

    for relative in EXPECTED_DOCS:
        path = repo / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty documentation file: {relative}")

    large_tracked = git_tracked_large_files(repo)
    if large_tracked:
        details = ", ".join(f"{name} ({size} bytes)" for name, size in large_tracked)
        raise AssertionError(f"Tracked files at or above 100 MB: {details}")

    return {
        "status": "PASS",
        "champion": champion["run_name"],
        "canonical_baseline": baseline["run_name"],
        "canonical_contract_validated": True,
        "strict_replay_claim": False,
        "tracked_files_over_100_mb": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if args.run_tests:
        run([sys.executable, "-m", "pytest", "-q"], cwd=repo)

    result = verify(repo)
    report_path = (
        repo
        / "reports"
        / "repository_finalization"
        / "release_contract_verification.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Release-contract verification PASS")


if __name__ == "__main__":
    main()
