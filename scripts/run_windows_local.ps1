param(
    [ValidateSet("10m", "20m", "32m")]
    [string]$Variant = "10m",

    [ValidateSet("quick", "full")]
    [string]$Profile = "quick",

    [string]$RepoPath = "C:\Users\bjw-0\Downloads\media-search-ranking-reliability-framework",
    [string]$ProjectDataRoot = "C:\Users\bjw-0\Downloads\Project_Data",

    [switch]$Install,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$datasetMap = @{
    "10m" = Join-Path $ProjectDataRoot "ml-10m"
    "20m" = Join-Path $ProjectDataRoot "ml-20m"
    "32m" = Join-Path $ProjectDataRoot "ml-32m"
}

$datasetPath = $datasetMap[$Variant]
if (-not (Test-Path $RepoPath)) {
    throw "Repo path not found: $RepoPath"
}
if (-not (Test-Path $datasetPath)) {
    throw "Dataset path not found: $datasetPath"
}

Set-Location $RepoPath
$venvPython = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.11 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv
    } else {
        throw "Python was not found on PATH."
    }
}

if ($Install) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e ".[dev,gpu,dashboard]"
}

$env:MOVIELENS_RAW_DIR = $datasetPath
$config = if ($Profile -eq "full") {
    "configs/pipeline_external_gpu_full.yaml"
} else {
    "configs/pipeline_external_gpu_quick.yaml"
}

Write-Host "Repo:      $RepoPath"
Write-Host "Dataset:   $datasetPath"
Write-Host "Variant:   $Variant"
Write-Host "Profile:   $Profile"
Write-Host "Config:    $config"

& $venvPython scripts/check_movielens_path.py $datasetPath --json-out artifacts/local_dataset_path_report.json
& $venvPython scripts/preflight_check.py

if (-not $SkipTests) {
    & $venvPython -m pytest -q
}

& $venvPython scripts/run_pipeline.py --config $config --mode movielens
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Run completed. Review:"
Write-Host "  artifacts/eval_summary.json"
Write-Host "  artifacts/launch_gate.json"
Write-Host "  reports/03_lambdarank_training_report.md"
Write-Host "  reports/07_launch_readiness_memo.md"
Write-Host "  reports/12_environment_and_runnability_report.md"
