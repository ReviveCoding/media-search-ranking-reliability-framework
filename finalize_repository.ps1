param(
    [string]$RepoPath = (Get-Location).Path,
    [switch]$ApplyCleanup,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path

$VenvPython = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Write-Host "[1/4] Synchronizing README and final documentation"
$FinalizeArgs = @(
    (Join-Path $RepoPath "scripts\finalize_repository.py"),
    "--repo", $RepoPath
)
if ($ApplyCleanup) {
    $FinalizeArgs += "--apply-cleanup"
}
& $Python @FinalizeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Repository finalization failed"
}

Write-Host "[2/4] Running release-contract verification"
$VerifyArgs = @(
    (Join-Path $RepoPath "scripts\verify_release_contract.py"),
    "--repo", $RepoPath
)
if (-not $SkipTests) {
    $VerifyArgs += "--run-tests"
}
& $Python @VerifyArgs
if ($LASTEXITCODE -ne 0) {
    throw "Release-contract verification failed"
}

Write-Host "[3/4] Showing generated finalization artifacts"
Get-ChildItem (Join-Path $RepoPath "reports\repository_finalization") |
    Format-Table Name, Length, LastWriteTime -AutoSize

Write-Host "[4/4] Finalization PASS"
Write-Host "Next: run .\verify_clean_checkout.ps1 for isolated GitHub-style verification."
