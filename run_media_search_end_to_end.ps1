[CmdletBinding()]
param(
    [string]$RepoPath = "C:\Users\bjw-0\Downloads\media-search-ranking-reliability-framework",
    [string]$DatasetPath = "C:\Users\bjw-0\Downloads\Project_Data\ml-10m",
    [switch]$RecreateVenv,
    [switch]$RunFull,
    [switch]$AllowCpuFallback,
    [switch]$SkipMonteCarlo,
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

function Write-Stage {
    param([int]$Number, [string]$Title)
    Write-Host ""
    Write-Host ("=" * 88) -ForegroundColor DarkCyan
    Write-Host ("STEP {0}: {1}" -f $Number, $Title) -ForegroundColor Cyan
    Write-Host ("=" * 88) -ForegroundColor DarkCyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$Description
    )

    # Always normalize arguments to an array. This makes zero-, one-, and many-argument
    # native command invocations safe under Set-StrictMode on Windows PowerShell 5.1.
    $SafeArguments = @($Arguments)

    Write-Host "`n> $Description" -ForegroundColor Yellow
    if ($SafeArguments.Count -gt 0) {
        Write-Host ("  {0} {1}" -f $Executable, ($SafeArguments -join " ")) -ForegroundColor DarkGray
        & $Executable @SafeArguments
    }
    else {
        Write-Host ("  {0}" -f $Executable) -ForegroundColor DarkGray
        & $Executable
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Required JSON file was not generated: $Path"
    }
    return Get-Content $Path -Raw | ConvertFrom-Json
}

Write-Stage 0 "Validate repository, dataset, Python, and NVIDIA driver"

if (-not (Test-Path -LiteralPath $RepoPath -PathType Container)) {
    throw "Repository folder was not found: $RepoPath"
}
if (-not (Test-Path -LiteralPath $DatasetPath -PathType Container)) {
    throw "MovieLens dataset folder was not found: $DatasetPath"
}

Set-Location -LiteralPath $RepoPath

$RequiredRepoFiles = @(
    "pyproject.toml",
    "configs\pipeline_gpu.yaml",
    "configs\pipeline_external_gpu_quick.yaml",
    "configs\pipeline_external_gpu_full.yaml",
    "scripts\run_pipeline.py",
    "scripts\preflight_check.py",
    "scripts\api_smoke_test.py"
)
$MissingRepoFiles = @($RequiredRepoFiles | Where-Object { -not (Test-Path (Join-Path $RepoPath $_)) })
if ($MissingRepoFiles.Count -gt 0) {
    throw "Repository is missing required v8 files: $($MissingRepoFiles -join ', ')"
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Invoke-Checked -Executable "nvidia-smi" -Description "Inspect NVIDIA GPU and driver"
} elseif (-not $AllowCpuFallback) {
    throw "nvidia-smi was not found. Install/update the NVIDIA driver, or rerun with -AllowCpuFallback."
} else {
    Write-Warning "nvidia-smi was not found. CPU fallback is allowed for this run."
}

$VenvPath = Join-Path $RepoPath ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Write-Stage 1 "Create the project-local virtual environment"

if ($RecreateVenv -and (Test-Path $VenvPath)) {
    Write-Host "Removing existing virtual environment: $VenvPath" -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

if (-not (Test-Path $VenvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Checked -Executable "py" -Arguments @("-3.11", "-m", "venv", ".venv") -Description "Create .venv with Python 3.11"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Invoke-Checked -Executable "python" -Arguments @("-m", "venv", ".venv") -Description "Create .venv with the available Python"
    } else {
        throw "Neither 'py' nor 'python' was found on PATH. Install 64-bit Python 3.11 first."
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "Virtual-environment Python was not created: $VenvPython"
}

# Activation makes subsequent console entry points available in this shell.
. (Join-Path $VenvPath "Scripts\Activate.ps1")
Invoke-Checked -Executable $VenvPython -Arguments @("--version") -Description "Confirm virtual-environment Python"

$PythonVersionText = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect Python version." }
if ([version]$PythonVersionText -lt [version]"3.10") {
    throw "Python >= 3.10 is required. Detected: $PythonVersionText"
}

Write-Stage 2 "Upgrade packaging tools and install CUDA-enabled PyTorch"

Invoke-Checked -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -Description "Upgrade pip, setuptools, and wheel"

# Official matched PyTorch 2.11 wheels for CUDA 12.8.
Invoke-Checked -Executable $VenvPython -Arguments @(
    "-m", "pip", "install", "--upgrade",
    "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
    "--index-url", "https://download.pytorch.org/whl/cu128"
) -Description "Install PyTorch 2.11 with CUDA 12.8 wheels"

Write-Stage 3 "Install framework, development, GPU, and dashboard dependencies"

Invoke-Checked -Executable $VenvPython -Arguments @("-m", "pip", "install", "-e", ".[dev,gpu,dashboard]") -Description "Install project extras in editable mode"
Invoke-Checked -Executable $VenvPython -Arguments @("-m", "pip", "check") -Description "Check dependency consistency"

Write-Stage 4 "Configure local cache and external MovieLens path"

$env:MOVIELENS_RAW_DIR = $DatasetPath
$env:HF_HOME = Join-Path $RepoPath ".cache\huggingface"
$env:TORCH_HOME = Join-Path $RepoPath ".cache\torch"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "42"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TORCH_HOME, (Join-Path $RepoPath "logs") | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $RepoPath "logs\end_to_end_$Timestamp.log"
Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    Write-Host "Repository: $RepoPath"
    Write-Host "MovieLens dataset: $DatasetPath"
    Write-Host "Hugging Face cache: $env:HF_HOME"
    Write-Host "Run transcript: $TranscriptPath"

    Write-Stage 5 "Verify CUDA, framework imports, and MovieLens layout"

    $CudaCode = @"
import json
import torch
payload = {
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_runtime': torch.version.cuda,
    'device_count': torch.cuda.device_count(),
    'device_name_0': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, indent=2))
if not payload['cuda_available']:
    raise SystemExit(2)
"@
    $CudaCode | & $VenvPython -
    $CudaExit = $LASTEXITCODE
    if (($CudaExit -ne 0) -and (-not $AllowCpuFallback)) {
        throw "PyTorch cannot access CUDA. Update the NVIDIA driver or reinstall a compatible PyTorch CUDA wheel."
    }
    if ($CudaExit -ne 0) {
        Write-Warning "CUDA verification failed, but -AllowCpuFallback was provided."
    }

    Invoke-Checked -Executable $VenvPython -Arguments @(
        "scripts\check_movielens_path.py", $DatasetPath,
        "--json-out", "artifacts\local_dataset_path_report.json"
    ) -Description "Resolve and validate the external MovieLens directory"

    Invoke-Checked -Executable $VenvPython -Arguments @("scripts\preflight_check.py") -Description "Generate environment, GPU, and dataset readiness reports"

    Write-Stage 6 "Run unit, component, loader, backend, and contract tests"

    Invoke-Checked -Executable $VenvPython -Arguments @("-m", "pytest", "-q") -Description "Run the full pytest suite"

    Write-Stage 7 "Run the strict GPU synthetic smoke pipeline"

    Invoke-Checked -Executable $VenvPython -Arguments @(
        "scripts\smoke_test.py",
        "--config", "configs\pipeline_gpu.yaml",
        "--mode", "demo"
    ) -Description "Run synthetic data processing, retrieval, LambdaRank training, inference, evaluation, and artifact checks"

    Write-Stage 8 "Run independent MovieLens-format smoke fixtures"

    Invoke-Checked -Executable $VenvPython -Arguments @("scripts\dataset_smoke_test.py", "--quick") -Description "Validate DAT and CSV MovieLens ingestion paths"

    Write-Stage 9 "Run the real MovieLens quick end-to-end pipeline"

    Invoke-Checked -Executable $VenvPython -Arguments @(
        "scripts\run_pipeline.py",
        "--config", "configs\pipeline_external_gpu_quick.yaml",
        "--mode", "movielens"
    ) -Description "Process MovieLens, train retrieval/ranking models, run inference, evaluate, and write reports"

    Write-Stage 10 "Run API inference and batch-search smoke tests"

    Invoke-Checked -Executable $VenvPython -Arguments @("scripts\api_smoke_test.py") -Description "Validate /ready, /search, /batch_search, /evaluate, and /launch_gate"

    $SampleInferencePath = Join-Path $RepoPath "artifacts\sample_inference.json"
    $SampleInferenceScript = Join-Path $RepoPath "artifacts\_sample_inference.py"
    @'
import json
from fastapi.testclient import TestClient
from media_search_reliability.api.app import app

client = TestClient(app)
client.get('/ready').raise_for_status()
response = client.post('/search', json={
    'query': 'funny sci-fi movie with robots',
    'top_k': 10,
    'preferred_genres': ['Comedy', 'Sci-Fi'],
})
response.raise_for_status()
payload = response.json()
with open('artifacts/sample_inference.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)
print(json.dumps({
    'query': payload['query'],
    'model_variant': payload['model_variant'],
    'latency_ms': payload['latency_ms'],
    'top_results': [
        {
            'movie_id': item['movie_id'],
            'title': item['title'],
            'score': item['score'],
            'calibrated_score': item['calibrated_score'],
        }
        for item in payload['results'][:5]
    ],
}, indent=2))
'@ | Set-Content -LiteralPath $SampleInferenceScript -Encoding UTF8
    Invoke-Checked -Executable $VenvPython -Arguments @($SampleInferenceScript) -Description "Generate a saved sample inference response"
    Remove-Item -LiteralPath $SampleInferenceScript -Force

    Write-Stage 11 "Run reproducibility and Monte Carlo reliability validation"

    Invoke-Checked -Executable $VenvPython -Arguments @("scripts\reproducibility_check.py") -Description "Verify cross-process deterministic replay"
    if (-not $SkipMonteCarlo) {
        Invoke-Checked -Executable $VenvPython -Arguments @(
            "scripts\monte_carlo_validate.py",
            "--trials-per-scenario", "1"
        ) -Description "Run the quick paired Monte Carlo stress audit"
    } else {
        Write-Warning "Monte Carlo validation was skipped."
    }

    Write-Stage 12 "Build source and wheel packages"

    if (-not $SkipBuild) {
        Invoke-Checked -Executable $VenvPython -Arguments @("-m", "build") -Description "Build source distribution and wheel"
    } else {
        Write-Warning "Package build was skipped."
    }

    if ($RunFull) {
        Write-Stage 13 "Run the full MovieLens GPU-oriented benchmark"

        Invoke-Checked -Executable $VenvPython -Arguments @(
            "scripts\run_pipeline.py",
            "--config", "configs\pipeline_external_gpu_full.yaml",
            "--mode", "movielens"
        ) -Description "Run the full external MovieLens training and evaluation profile"

        # Re-run API inference against the final full-run bundles.
        Invoke-Checked -Executable $VenvPython -Arguments @("scripts\api_smoke_test.py") -Description "Validate API against full-run model bundles"
    }

    Write-Stage 14 "Summarize model backends, quality, latency, and launch decision"

    $Summary = Read-JsonFile -Path (Join-Path $RepoPath "artifacts\eval_summary.json")
    $Gate = Read-JsonFile -Path (Join-Path $RepoPath "artifacts\launch_gate.json")
    $Environment = Read-JsonFile -Path (Join-Path $RepoPath "artifacts\environment_report.json")

    [pscustomobject]@{
        DataSource          = $Summary.data_source
        LaunchDecision      = $Summary.launch_decision
        DenseBackend        = $Summary.dense_backend
        VectorBackend       = $Summary.vector_index_backend
        RankerBackend       = $Summary.ranker_backend
        Calibration         = $Summary.calibration_method
        NDCG10              = $Summary.metrics.ndcg_at_10
        MRR10               = $Summary.metrics.mrr_at_10
        RecallEfficiency10  = $Summary.metrics.recall_efficiency_at_10
        ECE                 = $Summary.metrics.ece
        P95LatencyMs        = $Summary.metrics.p95_latency_ms
        TorchCudaAvailable  = $Environment.gpu.torch_cuda_available
        TorchCudaDevice     = $Environment.gpu.torch_cuda_device_name_0
        GateDecision        = $Gate.decision
    } | Format-List

    Write-Host ""
    Write-Host "END-TO-END RUN COMPLETED" -ForegroundColor Green
    Write-Host "Review these outputs:" -ForegroundColor Green
    Write-Host "  artifacts\environment_report.json"
    Write-Host "  artifacts\eval_summary.json"
    Write-Host "  artifacts\launch_gate.json"
    Write-Host "  artifacts\sample_inference.json"
    Write-Host "  reports\03_lambdarank_training_report.md"
    Write-Host "  reports\04_ablation_report.md"
    Write-Host "  reports\05_quality_latency_frontier.md"
    Write-Host "  reports\06_slice_reliability_report.md"
    Write-Host "  reports\07_launch_readiness_memo.md"
    Write-Host "  reports\12_environment_and_runnability_report.md"
    Write-Host "  reports\14_monte_carlo_validation_report.md"
    Write-Host ""
    Write-Host "To start the API in a new activated PowerShell window:" -ForegroundColor Cyan
    Write-Host "  cd `"$RepoPath`""
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  media-search-api --host 127.0.0.1 --port 8000"
    Write-Host ""
    Write-Host "To start the Streamlit dashboard in another activated window:" -ForegroundColor Cyan
    Write-Host "  cd `"$RepoPath`""
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  streamlit run dashboards\streamlit_app.py"
}
finally {
    try { Stop-Transcript | Out-Null } catch { }
}
