param(
    [string]$RepoPath = (Get-Location).Path,
    [switch]$KeepTemp
)

$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("media-search-clean-checkout-" + [guid]::NewGuid().ToString("N"))
$CheckoutPath = Join-Path $TempRoot "repo"
New-Item -ItemType Directory -Path $CheckoutPath -Force | Out-Null

try {
    Write-Host "[1/6] Building an isolated checkout from committed HEAD"
    Push-Location $RepoPath
    try {
        $GitAvailable = $null -ne (Get-Command git -ErrorAction SilentlyContinue)
        if ($GitAvailable -and (Test-Path ".git")) {
            $Status = git status --porcelain
            if ($LASTEXITCODE -ne 0) {
                throw "git status failed"
            }
            if ($Status) {
                Write-Host $Status
                throw "Working tree is not clean. Commit the finalized repository before clean-checkout verification."
            }

            $ArchivePath = Join-Path $TempRoot "head.zip"
            git archive --format=zip --output=$ArchivePath HEAD
            if ($LASTEXITCODE -ne 0) {
                throw "git archive HEAD failed"
            }
            Expand-Archive -LiteralPath $ArchivePath -DestinationPath $CheckoutPath -Force
        } else {
            Write-Warning "No Git repository detected; using a filesystem copy instead of committed HEAD."
            Get-ChildItem $RepoPath -Force |
                Where-Object { $_.Name -notin @(".git", ".venv", "venv") } |
                Copy-Item -Destination $CheckoutPath -Recurse -Force
        }
    } finally {
        Pop-Location
    }

    Write-Host "[2/6] Creating a clean virtual environment"
    $BasePython = "python"
    & $BasePython -m venv (Join-Path $CheckoutPath ".venv")
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    $Python = Join-Path $CheckoutPath ".venv\Scripts\python.exe"

    Write-Host "[3/6] Installing declared dependencies"
    Push-Location $CheckoutPath
    try {
        & $Python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

        if (Test-Path "requirements.txt") {
            & $Python -m pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) { throw "requirements installation failed" }
        }

        $InstallEditable = $false
        if (Test-Path "setup.py") { $InstallEditable = $true }
        if (Test-Path "setup.cfg") { $InstallEditable = $true }
        if (Test-Path "pyproject.toml") {
            $Pyproject = Get-Content "pyproject.toml" -Raw
            if ($Pyproject -match "(?m)^\[project\]\s*$") {
                $InstallEditable = $true
            }
        }

        if ($InstallEditable) {
            & $Python -m pip install -e .
            if ($LASTEXITCODE -ne 0) { throw "editable project installation failed" }
        } elseif (-not (Test-Path "requirements.txt")) {
            & $Python -m pip install pytest
            if ($LASTEXITCODE -ne 0) { throw "pytest installation failed" }
        }

        Write-Host "[4/6] Running compile and test checks"
        if (Test-Path "src") {
            & $Python -m compileall -q src
            if ($LASTEXITCODE -ne 0) { throw "src compile check failed" }
        }
        if (Test-Path "scripts") {
            & $Python -m compileall -q scripts
            if ($LASTEXITCODE -ne 0) { throw "scripts compile check failed" }
        }
        & $Python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "pytest failed in clean checkout" }

        Write-Host "[5/6] Verifying frozen release contract"
        & $Python scripts\verify_release_contract.py --repo $CheckoutPath
        if ($LASTEXITCODE -ne 0) { throw "release contract failed in clean checkout" }

        Write-Host "[6/6] CLEAN CHECKOUT PASS"
        Write-Host "Verified committed HEAD in: $CheckoutPath"
    } finally {
        Pop-Location
    }
}
finally {
    if (-not $KeepTemp -and (Test-Path $TempRoot)) {
        Remove-Item $TempRoot -Recurse -Force
    } elseif ($KeepTemp) {
        Write-Host "Kept temporary checkout at: $CheckoutPath"
    }
}
