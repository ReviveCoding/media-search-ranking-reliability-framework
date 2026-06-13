[CmdletBinding()]
param(
    [string]$RepoPath = "C:\Users\bjw-0\Downloads\media-search-multimodal-discovery-reliability-framework",
    [string]$MovieLensPath = "C:\Users\bjw-0\Downloads\Project_Data\ml-10m",
    [string]$MovieLens20MPath = "C:\Users\bjw-0\Downloads\Project_Data\ml-20m",
    [string]$TagGenomePath = "C:\Users\bjw-0\Downloads\Project_Data\genome_2021",
    [string]$ImdbPath = "C:\Users\bjw-0\Downloads\Project_Data\IMDb Non-Commercial Datasets",
    [switch]$RunFull,
    [switch]$SkipTagGenome,
    [switch]$SkipIMDb,
    [switch]$SkipMonteCarlo,
    [switch]$SkipBuild,
    [switch]$SkipPatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 92) -ForegroundColor DarkCyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("=" * 92) -ForegroundColor DarkCyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory=$true)][string]$Description
    )
    $SafeArguments = @($Arguments)
    Write-Host "`n> $Description" -ForegroundColor Yellow
    Write-Host ("  {0} {1}" -f $Executable, ($SafeArguments -join " ")) -ForegroundColor DarkGray
    if ($SafeArguments.Count -gt 0) { & $Executable @SafeArguments } else { & $Executable }
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

function Invoke-Optional {
    param(
        [Parameter(Mandatory=$true)][scriptblock]$Action,
        [Parameter(Mandatory=$true)][string]$Description
    )
    try {
        & $Action | Out-Host
        return $true
    }
    catch {
        Write-Warning "$Description failed and will be recorded as optional: $($_.Exception.Message)"
        return $false
    }
}

function Copy-RunSnapshot {
    param([string]$Name)
    $Target = Join-Path $RunRoot $Name
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    foreach ($Relative in @(
        "artifacts\eval_summary.json",
        "artifacts\launch_gate.json",
        "artifacts\ablation_metrics.csv",
        "artifacts\slice_metrics.csv",
        "artifacts\latency_by_variant.csv",
        "artifacts\sample_inference.json",
        "reports\03_lambdarank_training_report.md",
        "reports\04_ablation_report.md",
        "reports\06_slice_reliability_report.md",
        "reports\07_launch_readiness_memo.md",
        "reports\26_metadata_enrichment_report.md"
    )) {
        $Source = Join-Path $RepoPath $Relative
        if (Test-Path -LiteralPath $Source) {
            $Leaf = Split-Path $Relative -Leaf
            Copy-Item -LiteralPath $Source -Destination (Join-Path $Target $Leaf) -Force
        }
    }
}

Write-Step "STEP 0: Validate repository, environment, and external datasets"
if (-not (Test-Path -LiteralPath $RepoPath -PathType Container)) { throw "Repo not found: $RepoPath" }
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Project virtual environment not found: $Python" }
if (-not (Test-Path -LiteralPath $MovieLensPath -PathType Container)) { throw "MovieLens 10M path not found: $MovieLensPath" }
$PatchRoot = Join-Path $PSScriptRoot "_upgrade_payload"
if (-not $SkipPatch -and -not (Test-Path -LiteralPath $PatchRoot -PathType Container)) {
    throw "Upgrade payload not found beside the script: $PatchRoot"
}

Set-Location $RepoPath
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $RepoPath "backups\quality_upgrade_$Timestamp"
$RunRoot = Join-Path $RepoPath "artifacts\quality_upgrade"
New-Item -ItemType Directory -Force -Path $BackupRoot, $RunRoot | Out-Null

Write-Host "Repo:        $RepoPath"
Write-Host "MovieLens:   $MovieLensPath"
Write-Host "Tag Genome:  $TagGenomePath"
Write-Host "IMDb:        $ImdbPath"
Write-Host "Python:      $Python"

Write-Step "STEP 1: Preserve current baseline and create a source backup"
foreach ($Item in @("src", "scripts", "tests", "configs", "pyproject.toml", "README.md")) {
    $Source = Join-Path $RepoPath $Item
    if (Test-Path -LiteralPath $Source) { Copy-Item -LiteralPath $Source -Destination $BackupRoot -Recurse -Force }
}
$BaselineDir = Join-Path $RunRoot "baseline_pre_upgrade"
New-Item -ItemType Directory -Force -Path $BaselineDir | Out-Null
if (Test-Path ".\artifacts\eval_summary.json") { Copy-Item ".\artifacts\eval_summary.json" (Join-Path $BaselineDir "eval_summary.json") -Force }
if (Test-Path ".\artifacts\launch_gate.json") { Copy-Item ".\artifacts\launch_gate.json" (Join-Path $BaselineDir "launch_gate.json") -Force }
Write-Host "Backup: $BackupRoot" -ForegroundColor Green

if (-not $SkipPatch) {
    Write-Step "STEP 2: Apply query, specialized-retrieval, personalization, and metadata-enrichment upgrade"
    $PatchFiles = @(Get-ChildItem -LiteralPath $PatchRoot -Recurse -File)
    foreach ($File in $PatchFiles) {
        $Relative = $File.FullName.Substring($PatchRoot.Length).TrimStart([char[]]@('\', '/'))
        $Destination = Join-Path $RepoPath $Relative
        New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
        Copy-Item -LiteralPath $File.FullName -Destination $Destination -Force
    }
    Write-Host "Applied $($PatchFiles.Count) upgrade files." -ForegroundColor Green
}

Write-Step "STEP 3: Reinstall editable package and run the complete test suite"
Invoke-Checked $Python @("-m", "pip", "install", "-e", ".[dev,gpu,dashboard]") "Install upgraded project in the existing virtual environment"
Invoke-Checked $Python @("-m", "pytest", "-q") "Run unit, leakage, backend-contract, API, and upgrade tests"

Write-Step "STEP 4: Run small synthetic regression before public-data training"
$env:MOVIELENS_RAW_DIR = $MovieLensPath
Remove-Item Env:TAG_GENOME_ENRICHMENT_PATH -ErrorAction SilentlyContinue
Remove-Item Env:IMDB_ENRICHMENT_PATH -ErrorAction SilentlyContinue
Invoke-Checked $Python @("scripts\run_pipeline.py", "--config", "configs\pipeline.yaml", "--mode", "demo") "Run upgraded synthetic end-to-end smoke"
Copy-RunSnapshot "synthetic_upgrade_smoke"

Write-Step "STEP 5: Run the core MovieLens 10M quality upgrade benchmark"
Invoke-Checked $Python @("scripts\run_pipeline.py", "--config", "configs\pipeline_external_gpu_quick.yaml", "--mode", "movielens") "Train and evaluate normalized-query + specialized-candidate LambdaRank"
Copy-RunSnapshot "core_upgrade_quick"
$CoreSummary = Join-Path $RunRoot "core_upgrade_quick\eval_summary.json"

Write-Step "STEP 6: Build optional Tag Genome enrichment and mapping audit"
$TagOutput = Join-Path $RepoPath "data\processed\tag_genome_enrichment.csv"
$TagReady = $false
if (-not $SkipTagGenome) {
    $TagCandidates = @($TagGenomePath, $MovieLens20MPath) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }
    foreach ($Candidate in $TagCandidates) {
        $Succeeded = Invoke-Optional -Description "Tag Genome adapter for $Candidate" -Action {
            Invoke-Checked $Python @(
                "scripts\build_tag_genome_enrichment.py",
                "--genome-dir", $Candidate,
                "--catalog", "data\processed\media_catalog.csv",
                "--output", "data\processed\tag_genome_enrichment.csv",
                "--report", "reports\24_tag_genome_mapping_report.json"
            ) "Build Tag Genome top-tag features and mapping report"
        }
        if ($Succeeded -and (Test-Path -LiteralPath $TagOutput)) { $TagReady = $true; break }
    }
}
if ($TagReady) {
    $env:TAG_GENOME_ENRICHMENT_PATH = $TagOutput
    Write-Host "Tag Genome enrichment ready: $TagOutput" -ForegroundColor Green
} else {
    Write-Warning "Tag Genome enrichment was not enabled. The core benchmark remains valid."
}

Write-Step "STEP 7: Build optional IMDb title/rating/runtime enrichment and mapping audit"
$ImdbOutput = Join-Path $RepoPath "data\processed\imdb_enrichment.csv"
$ImdbReady = $false
if (-not $SkipIMDb -and (Test-Path -LiteralPath $ImdbPath -PathType Container)) {
    $ImdbReady = Invoke-Optional -Description "IMDb metadata adapter" -Action {
        Invoke-Checked $Python @(
            "scripts\build_imdb_enrichment.py",
            "--imdb-dir", $ImdbPath,
            "--catalog", "data\processed\media_catalog.csv",
            "--output", "data\processed\imdb_enrichment.csv",
            "--report", "reports\25_imdb_mapping_report.json"
        ) "Build IMDb exact title-year metadata enrichment"
    }
}
if ($ImdbReady -and (Test-Path -LiteralPath $ImdbOutput)) {
    $env:IMDB_ENRICHMENT_PATH = $ImdbOutput
    Write-Host "IMDb enrichment ready: $ImdbOutput" -ForegroundColor Green
} else {
    $ImdbReady = $false
    Write-Warning "IMDb enrichment was not enabled. This does not invalidate the core benchmark."
}

$EnrichedSummary = $null
if ($TagReady -or $ImdbReady) {
    Write-Step "STEP 8: Run the metadata-enriched MovieLens 10M benchmark"
    Invoke-Checked $Python @("scripts\run_pipeline.py", "--config", "configs\pipeline_external_gpu_quick.yaml", "--mode", "movielens") "Train and evaluate the enriched ranking model"
    Copy-RunSnapshot "metadata_enriched_quick"
    $EnrichedSummary = Join-Path $RunRoot "metadata_enriched_quick\eval_summary.json"
} else {
    Write-Step "STEP 8: Metadata benchmark skipped because no enrichment adapter produced a usable file"
}

Write-Step "STEP 9: Compare baseline, core upgrade, and enriched results"
$BaselineSummary = Join-Path $BaselineDir "eval_summary.json"
if (-not (Test-Path -LiteralPath $BaselineSummary)) { $BaselineSummary = $CoreSummary }
$CompareArgs = @(
    "scripts\compare_upgrade_results.py",
    "--baseline", $BaselineSummary,
    "--core", $CoreSummary,
    "--output", "reports\27_quality_upgrade_comparison.md"
)
if ($EnrichedSummary -and (Test-Path -LiteralPath $EnrichedSummary)) { $CompareArgs += @("--enriched", $EnrichedSummary) }
Invoke-Checked $Python $CompareArgs "Generate the before/after ablation comparison report"

Write-Step "STEP 10: Validate serving, deterministic replay, and reliability stress behavior"
Invoke-Checked $Python @("scripts\api_smoke_test.py") "Run API inference and batch-search smoke tests"
Invoke-Checked $Python @("scripts\reproducibility_check.py") "Verify cross-process deterministic replay"
if (-not $SkipMonteCarlo) {
    Invoke-Checked $Python @("scripts\monte_carlo_validate.py", "--trials-per-scenario", "1") "Run paired Monte Carlo stress validation"
}

if ($RunFull) {
    Write-Step "STEP 11: Run the full improved MovieLens 10M benchmark"
    Invoke-Checked $Python @("scripts\run_pipeline.py", "--config", "configs\pipeline_external_gpu_full.yaml", "--mode", "movielens") "Run the full query/candidate/training configuration"
    Copy-RunSnapshot "final_full_benchmark"
}

if (-not $SkipBuild) {
    Write-Step "STEP 12: Build release artifacts"
    Invoke-Checked $Python @("-m", "build") "Build source distribution and wheel"
}

Write-Step "FINAL SUMMARY"
$FinalSummaryPath = if ($RunFull) { Join-Path $RunRoot "final_full_benchmark\eval_summary.json" } elseif ($EnrichedSummary) { $EnrichedSummary } else { $CoreSummary }
if (Test-Path -LiteralPath $FinalSummaryPath) {
    $Summary = Get-Content -LiteralPath $FinalSummaryPath -Raw | ConvertFrom-Json
    [pscustomobject]@{
        DataSource             = $Summary.data_source
        LaunchDecision         = $Summary.launch_decision
        DenseBackend           = $Summary.dense_backend
        VectorBackend          = $Summary.vector_index_backend
        RankerBackend          = $Summary.ranker_backend
        NDCG10                 = $Summary.metrics.ndcg_at_10
        MRR10                  = $Summary.metrics.mrr_at_10
        RecallEfficiency10     = $Summary.metrics.recall_efficiency_at_10
        SimilarToNDCG10        = $Summary.query_type_metrics.similar_to.ndcg_at_10
        PersonalizedNDCG10     = $Summary.query_type_metrics.personalized.ndcg_at_10
        MoodDecadeNDCG10       = $Summary.query_type_metrics.mood_decade.ndcg_at_10
        RankerLiftVsHybrid     = $Summary.ranker_ndcg_lift_vs_hybrid
        TagGenomeApplied       = $Summary.enrichment_diagnostics.tag_genome_applied
        IMDbApplied            = $Summary.enrichment_diagnostics.imdb_applied
    } | Format-List
}
Write-Host "`nQUALITY UPGRADE PIPELINE COMPLETED" -ForegroundColor Green
Write-Host "Comparison report: $RepoPath\reports\27_quality_upgrade_comparison.md"
Write-Host "Run snapshots:      $RunRoot"
Write-Host "Source backup:      $BackupRoot"
