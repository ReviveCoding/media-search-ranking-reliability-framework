param(
    [string]$RepoPath = "C:\Users\bjw-0\Downloads\media-search-ranking-reliability-framework"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location $RepoPath
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python was not found: $Python"
}

& $Python ".\scripts\aggregate_frozen_movielens_results.py"
if ($LASTEXITCODE -ne 0) {
    throw "Frozen-result aggregation failed"
}

Write-Host "Frozen result aggregation PASS" -ForegroundColor Green
