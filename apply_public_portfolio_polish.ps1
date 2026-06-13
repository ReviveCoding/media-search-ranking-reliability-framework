param(
    [string]$RepoPath = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path

$VenvPython = Join-Path $RepoPath ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Write-Host "[1/4] Applying public portfolio polish"
& $Python (Join-Path $RepoPath "scripts\apply_public_portfolio_polish.py") --repo $RepoPath
if ($LASTEXITCODE -ne 0) { throw "Public portfolio polish failed" }

Write-Host "[2/4] Running full tests"
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed after public portfolio polish" }

Write-Host "[3/4] Verifying public portfolio contract"
& $Python (Join-Path $RepoPath "scripts\verify_public_portfolio.py") --repo $RepoPath
if ($LASTEXITCODE -ne 0) { throw "Public portfolio verification failed" }

Write-Host "[4/4] PUBLIC PORTFOLIO POLISH PASS"
Write-Host "Review README.md and docs\PUBLIC_RELEASE_CHECKLIST.md before committing."
