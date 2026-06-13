from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


README_START = "<!-- FINAL_RESULTS_START -->"
README_END = "<!-- FINAL_RESULTS_END -->"
GITIGNORE_START = "# BEGIN AUTO FINALIZATION"
GITIGNORE_END = "# END AUTO FINALIZATION"

EXPECTED_STRICT_RUNS = {
    "core_champion_replay",
    "tag_genome_feature_only",
    "imdb_feature_only",
    "combined_feature_only",
}

SAFE_CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "htmlcov",
    "build",
    "dist",
}

SAFE_CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}
SAFE_CACHE_FILE_NAMES = {".coverage", "coverage.xml", ".DS_Store", "Thumbs.db"}

SKIP_SCAN_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
}


@dataclass(frozen=True)
class ReleaseData:
    summary: dict
    diagnostics: dict
    champion: dict
    baseline: dict


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def load_release_data(repo: Path) -> ReleaseData:
    artifact_dir = repo / "artifacts" / "frozen_quality_ablation"
    summary = load_json(artifact_dir / "frozen_ablation_summary.json")
    diagnostics = load_json(artifact_dir / "frozen_replay_diagnostics.json")

    champion = summary.get("champion")
    baseline = summary.get("canonical_core_baseline")
    if not isinstance(champion, dict) or not isinstance(baseline, dict):
        raise AssertionError("Summary must contain champion and canonical_core_baseline objects")

    return ReleaseData(
        summary=summary,
        diagnostics=diagnostics,
        champion=champion,
        baseline=baseline,
    )


def validate_release_contract(
    data: ReleaseData,
    expected_champion: str,
    expected_baseline: str,
) -> None:
    champion_name = data.champion.get("run_name")
    baseline_name = data.baseline.get("run_name")
    if champion_name != expected_champion:
        raise AssertionError(
            f"Expected champion {expected_champion!r}, found {champion_name!r}"
        )
    if baseline_name != expected_baseline:
        raise AssertionError(
            f"Expected baseline {expected_baseline!r}, found {baseline_name!r}"
        )
    if data.champion.get("eligible_for_promotion") is not True:
        raise AssertionError("Champion is not eligible_for_promotion")
    if data.diagnostics.get("canonical_contract_validated") is not True:
        raise AssertionError("Canonical frozen contract is not validated")
    if data.diagnostics.get("strict_replay_claim") is not False:
        raise AssertionError("Expected strict_replay_claim=false for legacy comparison")
    if data.diagnostics.get("status") != "legacy_not_strictly_comparable":
        raise AssertionError("Unexpected replay diagnostics status")

    strict_runs = set(data.summary.get("strict_contract_runs", []))
    if strict_runs != EXPECTED_STRICT_RUNS:
        raise AssertionError(
            f"Strict contract run set differs: {sorted(strict_runs)}"
        )

    champion_ndcg = float(data.champion["ndcg_at_10"])
    baseline_ndcg = float(data.baseline["ndcg_at_10"])
    if champion_ndcg < baseline_ndcg:
        raise AssertionError("Champion NDCG@10 is below canonical baseline")


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


def metric_rows(data: ReleaseData) -> list[tuple[str, str, float, float]]:
    keys = [
        ("NDCG@10", "ndcg_at_10"),
        ("MRR@10", "mrr_at_10"),
        ("Recall efficiency@10", "recall_efficiency_at_10"),
        ("Hit rate@10", "hit_rate_at_10"),
        ("Genre/tag NDCG", "genre_tag_ndcg"),
        ("Mood/decade NDCG", "mood_decade_ndcg"),
        ("Personalized NDCG", "personalized_ndcg"),
        ("Similar-to NDCG", "similar_to_ndcg"),
        ("Visual-query NDCG", "visual_query_ndcg"),
        ("ECE", "ece"),
        ("p95 latency (ms)", "p95_latency_ms"),
    ]
    rows = []
    for label, key in keys:
        base = float(data.baseline[key])
        champ = float(data.champion[key])
        rows.append((label, key, base, champ))
    return rows


def markdown_metric_table(data: ReleaseData) -> str:
    lines = [
        "| Metric | Canonical baseline | Promoted champion | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for label, _, base, champ in metric_rows(data):
        change = pct_change(champ, base)
        lines.append(f"| {label} | {base:.6f} | {champ:.6f} | {change:+.2f}% |")
    return "\n".join(lines)


def readme_block(data: ReleaseData) -> str:
    champion = data.champion
    baseline = data.baseline
    ndcg_change = pct_change(float(champion["ndcg_at_10"]), float(baseline["ndcg_at_10"]))
    return f"""{README_START}
## Final frozen benchmark result

- **Promoted champion:** `{champion["run_name"]}`
- **Canonical baseline:** `{baseline["run_name"]}`
- **NDCG@10:** `{float(baseline["ndcg_at_10"]):.6f}` → `{float(champion["ndcg_at_10"]):.6f}` ({ndcg_change:+.2f}%)
- **Frozen contract:** validated for the canonical baseline and enrichment variants
- **Legacy replay disclosure:** `legacy_not_strictly_comparable`; no strict GPU replay claim
- **Launch decision:** `{champion.get("launch_decision", "UNKNOWN")}`

See [final results](docs/FINAL_RESULTS.md), [reproducibility](docs/REPRODUCIBILITY.md), and [claim boundaries](docs/CLAIM_BOUNDARIES.md).
{README_END}"""


def update_marked_block(path: Path, start: str, end: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        updated = before.rstrip() + "\n\n" + block + after
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def final_results_document(data: ReleaseData) -> str:
    champion = data.champion
    baseline = data.baseline
    ndcg_delta = float(champion["ndcg_at_10"]) - float(baseline["ndcg_at_10"])
    ndcg_pct = pct_change(float(champion["ndcg_at_10"]), float(baseline["ndcg_at_10"]))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return f"""# Final Frozen MovieLens Ranking Results

Generated from the committed benchmark artifacts on {generated}.

## Decision

- Promoted champion: `{champion["run_name"]}`
- Canonical promotion baseline: `{baseline["run_name"]}`
- Champion NDCG@10: `{float(champion["ndcg_at_10"]):.6f}`
- Absolute NDCG@10 improvement: `{ndcg_delta:+.6f}`
- Relative NDCG@10 improvement: `{ndcg_pct:+.2f}%`
- Promotion eligibility: `{str(champion.get("eligible_for_promotion")).lower()}`
- Launch decision: `{champion.get("launch_decision", "UNKNOWN")}`

The promoted model is the best eligible run under the current frozen-contract comparison. The higher legacy `core_compact` score is retained only as profile-selection context and is not treated as a directly comparable promotion candidate.

## Canonical comparison

{markdown_metric_table(data)}

## Contract interpretation

The canonical manifest and query-group split were validated across:

- `core_champion_replay`
- `tag_genome_feature_only`
- `imdb_feature_only`
- `combined_feature_only`

Legacy profile runs are preserved for historical context. They do not support a strict GPU replay claim because their configuration fingerprints and, in one case, split contract differ from the canonical replay.

## Promotion rule

`{data.summary.get("promotion_rule", "Not recorded")}`

## Source artifacts

- `artifacts/frozen_quality_ablation/frozen_ablation_results.csv`
- `artifacts/frozen_quality_ablation/frozen_ablation_summary.json`
- `artifacts/frozen_quality_ablation/frozen_replay_diagnostics.json`
- `reports/frozen_quality_ablation/29_frozen_movielens_ablation.md`
"""


def reproducibility_document(data: ReleaseData) -> str:
    return """# Reproducibility and Verification

## Supported environment

- Python 3.11
- Windows PowerShell for the repository wrappers
- CPU or CUDA-capable GPU, depending on the selected ranking configuration
- Public MovieLens benchmark inputs and locally generated artifacts

## Fast verification

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\aggregate_frozen_quality_results.ps1
.\\finalize_repository.ps1
```

The aggregation command reuses completed variant summaries. It does not retrain the ranking models.

## Clean-checkout verification

```powershell
.\\verify_clean_checkout.ps1
```

This command copies publishable repository files into an isolated temporary directory, creates a fresh virtual environment, installs declared dependencies, runs compilation checks, runs the full test suite, and verifies the frozen release contract.

Use the following command to retain the isolated directory for manual inspection:

```powershell
.\\verify_clean_checkout.ps1 -KeepTemp
```

## Frozen comparison contract

The canonical promotion comparison requires exact agreement on the frozen manifest and query-group split. GPU metric replay against older profile-selection runs is diagnostic only because those runs do not share a fully matching configuration fingerprint.

## Expected final state

- Test suite passes
- Frozen aggregation passes
- Canonical contract is validated
- Promoted champion is `combined_feature_only`
- Canonical baseline is `core_champion_replay`
- Legacy strict replay claim remains disabled
"""


def claim_boundaries_document(data: ReleaseData) -> str:
    champion = data.champion
    baseline = data.baseline
    change = pct_change(float(champion["ndcg_at_10"]), float(baseline["ndcg_at_10"]))
    return f"""# Claim Boundaries

## Supported claims

- Built a runnable multimodal media-search and ranking-quality framework with frozen benchmark contracts, ranking evaluation, slice guardrails, calibration, latency reporting, and automated release checks.
- Promoted `combined_feature_only` over the canonical `core_champion_replay` baseline under the same frozen manifest and query-group split.
- Improved NDCG@10 from `{float(baseline["ndcg_at_10"]):.6f}` to `{float(champion["ndcg_at_10"]):.6f}`, a `{change:+.2f}%` relative improvement in the canonical comparison.
- Preserved legacy profile results as historical context while explicitly disabling a strict replay claim when configuration fingerprints do not match.
- Validated repository behavior through automated tests, artifact-contract checks, and an isolated clean-checkout workflow.

## Claims that should not be made

- Do not claim that `core_compact` and `core_champion_replay` are bit-for-bit GPU reproductions.
- Do not describe the legacy score difference as ordinary floating-point noise.
- Do not claim production deployment or online business impact.
- Do not claim that `ITERATE` means launch-ready.
- Do not state a latency improvement as causal unless it is confirmed through repeated, controlled benchmarking on identical hardware and workload.
- Do not imply proprietary user, streaming-platform, or company data.

## Recommended résumé wording

“Built a reproducible multimodal media-search ranking framework with frozen data/split contracts, LambdaRank evaluation, slice-regression guardrails, calibration and latency diagnostics; promoted a combined metadata feature variant that improved canonical NDCG@10 by {change:.2f}% while preserving explicit GPU replay and launch-readiness boundaries.”
"""


def gitignore_block() -> str:
    return f"""{GITIGNORE_START}
# Python environments and caches
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
coverage.xml
htmlcov/
.ipynb_checkpoints/

# Packaging/build output
build/
dist/
*.egg-info/

# OS/editor noise
.DS_Store
Thumbs.db
.vscode/
.idea/

# Local heavyweight caches and model checkpoints
data/cache/
data/raw/
artifacts/**/checkpoints/
*.pt
*.pth
*.onnx

# Temporary patch bundles
*hotfix*.zip
*hotfix*/
{GITIGNORE_END}"""


def iter_paths(repo: Path) -> Iterable[Path]:
    for path in repo.rglob("*"):
        try:
            relative_parts = path.relative_to(repo).parts
        except ValueError:
            continue
        if any(part in SKIP_SCAN_PARTS for part in relative_parts):
            continue
        yield path


def cleanup_candidates(repo: Path) -> tuple[list[Path], list[Path]]:
    dirs: list[Path] = []
    files: list[Path] = []
    for path in iter_paths(repo):
        if path.is_dir() and path.name in SAFE_CACHE_DIR_NAMES:
            dirs.append(path)
        elif path.is_file() and (
            path.name in SAFE_CACHE_FILE_NAMES or path.suffix in SAFE_CACHE_FILE_SUFFIXES
        ):
            files.append(path)
    return sorted(set(dirs)), sorted(set(files))


def apply_cleanup(repo: Path, dirs: list[Path], files: list[Path]) -> None:
    for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        if path.exists():
            shutil.rmtree(path)
    for path in files:
        if path.exists():
            path.unlink()


def write_cleanup_report(repo: Path, dirs: list[Path], files: list[Path], applied: bool) -> Path:
    report_dir = repo / "reports" / "repository_finalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "cleanup_candidates.txt"
    lines = [
        f"cleanup_applied={str(applied).lower()}",
        "",
        "[directories]",
        *[str(p.relative_to(repo)) for p in dirs],
        "",
        "[files]",
        *[str(p.relative_to(repo)) for p in files],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_large_file_report(repo: Path) -> tuple[Path, list[dict]]:
    report_dir = repo / "reports" / "repository_finalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "large_files.csv"
    rows: list[dict] = []
    for candidate in iter_paths(repo):
        if not candidate.is_file():
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size >= 10 * 1024 * 1024:
            rows.append(
                {
                    "path": str(candidate.relative_to(repo)),
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 3),
                    "over_50_mb": size >= 50 * 1024 * 1024,
                    "over_100_mb": size >= 100 * 1024 * 1024,
                }
            )
    rows.sort(key=lambda row: row["size_bytes"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "size_mb", "over_50_mb", "over_100_mb"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def write_finalization_summary(
    repo: Path,
    data: ReleaseData,
    cleanup_applied: bool,
    cleanup_count: int,
    large_files: list[dict],
) -> Path:
    report_dir = repo / "reports" / "repository_finalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "finalization_summary.json"
    payload = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion": data.champion.get("run_name"),
        "canonical_baseline": data.baseline.get("run_name"),
        "champion_ndcg_at_10": float(data.champion["ndcg_at_10"]),
        "baseline_ndcg_at_10": float(data.baseline["ndcg_at_10"]),
        "canonical_contract_validated": data.diagnostics.get(
            "canonical_contract_validated"
        ),
        "strict_replay_claim": data.diagnostics.get("strict_replay_claim"),
        "cleanup_applied": cleanup_applied,
        "cleanup_candidate_count": cleanup_count,
        "files_over_50_mb": sum(bool(row["over_50_mb"]) for row in large_files),
        "files_over_100_mb": sum(bool(row["over_100_mb"]) for row in large_files),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--apply-cleanup", action="store_true")
    parser.add_argument("--expected-champion", default="combined_feature_only")
    parser.add_argument("--expected-baseline", default="core_champion_replay")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        raise NotADirectoryError(repo)

    data = load_release_data(repo)
    validate_release_contract(
        data,
        expected_champion=args.expected_champion,
        expected_baseline=args.expected_baseline,
    )

    docs_dir = repo / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "FINAL_RESULTS.md").write_text(
        final_results_document(data), encoding="utf-8"
    )
    (docs_dir / "REPRODUCIBILITY.md").write_text(
        reproducibility_document(data), encoding="utf-8"
    )
    (docs_dir / "CLAIM_BOUNDARIES.md").write_text(
        claim_boundaries_document(data), encoding="utf-8"
    )

    readme_path = repo / "README.md"
    update_marked_block(
        readme_path,
        README_START,
        README_END,
        readme_block(data),
    )

    gitignore_path = repo / ".gitignore"
    update_marked_block(
        gitignore_path,
        GITIGNORE_START,
        GITIGNORE_END,
        gitignore_block(),
    )

    dirs, files = cleanup_candidates(repo)
    write_cleanup_report(repo, dirs, files, args.apply_cleanup)
    if args.apply_cleanup:
        apply_cleanup(repo, dirs, files)

    _, large_files = write_large_file_report(repo)
    summary_path = write_finalization_summary(
        repo,
        data,
        cleanup_applied=args.apply_cleanup,
        cleanup_count=len(dirs) + len(files),
        large_files=large_files,
    )

    print(f"Champion: {data.champion['run_name']}")
    print(f"Canonical baseline: {data.baseline['run_name']}")
    print(f"README synchronized: {readme_path}")
    print(f"Finalization summary: {summary_path}")
    print("Repository documentation synchronization PASS")


if __name__ == "__main__":
    main()
