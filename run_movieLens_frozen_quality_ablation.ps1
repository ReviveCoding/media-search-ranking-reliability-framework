[CmdletBinding()]
param(
    [string]$RepoPath = "C:\Users\bjw-0\Downloads\media-search-ranking-reliability-framework",
    [string]$MovieLensPath = "C:\Users\bjw-0\Downloads\Project_Data\ml-10m",
    [string]$TagGenomeEnrichment = "C:\Users\bjw-0\Downloads\media-search-ranking-reliability-framework\data\processed\tag_genome_enrichment.csv",
    [string]$ImdbEnrichment = "C:\Users\bjw-0\Downloads\media-search-ranking-reliability-framework\data\processed\imdb_enrichment.csv",
    [switch]$SkipRankerTuning,
    [switch]$RunFull,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $RepoPath
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment Python not found: $Python" }
if (-not (Test-Path $MovieLensPath)) { throw "MovieLens directory not found: $MovieLensPath" }

Remove-Item Env:TAG_GENOME_ENRICHMENT_PATH -ErrorAction SilentlyContinue
Remove-Item Env:IMDB_ENRICHMENT_PATH -ErrorAction SilentlyContinue

Write-Host "[1/4] Running regression tests" -ForegroundColor Cyan
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

$Config = if ($RunFull) { "configs\pipeline_external_gpu_full.yaml" } else { "configs\pipeline_external_gpu_quick.yaml" }
$ManifestDir = if ($RunFull) { "data\benchmarks\ml10m_frozen_v1_full" } else { "data\benchmarks\ml10m_frozen_v1" }
$OutputRoot = if ($RunFull) { "artifacts\frozen_quality_ablation_full" } else { "artifacts\frozen_quality_ablation" }
$ReportsRoot = if ($RunFull) { "reports\frozen_quality_ablation_full" } else { "reports\frozen_quality_ablation" }
$Args = @(
    "scripts\run_frozen_movielens_ablation.py",
    "--config", $Config,
    "--movielens-dir", $MovieLensPath,
    "--manifest-dir", $ManifestDir,
    "--output-root", $OutputRoot,
    "--reports-root", $ReportsRoot,
    "--num-queries", $(if ($RunFull) { "1000" } else { "500" }),
    "--candidates-per-query", $(if ($RunFull) { "120" } else { "100" })
)
if (Test-Path $TagGenomeEnrichment) { $Args += @("--tag-genome-enrichment", $TagGenomeEnrichment) }
if (Test-Path $ImdbEnrichment) { $Args += @("--imdb-enrichment", $ImdbEnrichment) }
if ($SkipRankerTuning) { $Args += "--skip-ranker-tuning" }
if ($Resume) { $Args += "--resume" }

Write-Host "[2/4] Running frozen MovieLens ablation" -ForegroundColor Cyan
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "Frozen ablation failed" }

Write-Host "[3/4] Validating frozen manifest and outputs" -ForegroundColor Cyan
$Required = @(
    (Join-Path $ManifestDir "manifest.json"),
    (Join-Path $ManifestDir "queries.csv"),
    (Join-Path $ManifestDir "judgments.csv"),
    (Join-Path $OutputRoot "frozen_ablation_results.csv"),
    (Join-Path $ReportsRoot "29_frozen_movielens_ablation.md")
)
$Missing = @($Required | Where-Object { -not (Test-Path $_) })
if ($Missing.Count -gt 0) { throw "Missing outputs: $($Missing -join ', ')" }

Write-Host "[4/4] Result summary" -ForegroundColor Cyan
Import-Csv (Join-Path $OutputRoot "frozen_ablation_results.csv") |
    Sort-Object { [double]$_.ndcg_at_10 } -Descending |
    Format-Table run_name, ranker_profile, ndcg_at_10, recall_efficiency_at_10, personalized_ndcg, similar_to_ndcg, launch_decision -AutoSize

Write-Host "Frozen MovieLens quality ablation PASS" -ForegroundColor Green
