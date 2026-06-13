# Local Windows run using the supplied dataset paths

## Recommended order

1. **MovieLens 10M**, quick profile, end-to-end GPU/CPU verification.
2. **MovieLens 10M**, full profile, primary benchmark.
3. **MovieLens 20M**, full profile, stronger final benchmark and Tag Genome-ready expansion.
4. **MovieLens 32M**, optional scale stress test only.

The current core pipeline does not consume BDD100K, banking, forecasting, fraud, dialogue, document, or financial datasets. It also does not silently consume IMDb, standalone Tag Genome, MSR-VTT, or YouTube trailer mappings. Those require explicit adapters and should remain separate extensions.

## One-command quick run

Open PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass

& "C:\Users\bjw-0\Downloads\media-search-multimodal-discovery-reliability-framework\scripts\run_windows_local.ps1" `
  -Variant 10m `
  -Profile quick `
  -Install
```

The helper uses:

```text
Repo:
C:\Users\bjw-0\Downloads\media-search-multimodal-discovery-reliability-framework

10M:
C:\Users\bjw-0\Downloads\Project_Data\ml-10m

20M:
C:\Users\bjw-0\Downloads\Project_Data\ml-20m

32M:
C:\Users\bjw-0\Downloads\Project_Data\ml-32m
```

It supports extracted files at the specified directory or within a nested child such as `ml-20m\ml-20m` or `ml-10m\ml-10M100K`.

## Manual commands

```powershell
cd "C:\Users\bjw-0\Downloads\media-search-multimodal-discovery-reliability-framework"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gpu,dashboard]"

$env:MOVIELENS_RAW_DIR = "C:\Users\bjw-0\Downloads\Project_Data\ml-10m"

media-search-check-path $env:MOVIELENS_RAW_DIR
python scripts/preflight_check.py
python -m pytest -q

python scripts/run_pipeline.py `
  --config configs/pipeline_external_gpu_quick.yaml `
  --mode movielens
```

Full benchmark:

```powershell
python scripts/run_pipeline.py `
  --config configs/pipeline_external_gpu_full.yaml `
  --mode movielens
```

Switch to MovieLens 20M without moving or copying data:

```powershell
$env:MOVIELENS_RAW_DIR = "C:\Users\bjw-0\Downloads\Project_Data\ml-20m"

python scripts/run_pipeline.py `
  --config configs/pipeline_external_gpu_full.yaml `
  --mode movielens
```

## GPU verification

Review:

```text
artifacts/environment_report.json
artifacts/eval_summary.json
```

Expected advanced path labels include:

```text
dense_backend: sentence-transformers...
ranker_backend: lightgbm-lambdarank-gpu
```

On Windows, FAISS availability depends on the installed build. A reported `sklearn-nearest-neighbors-fallback` is acceptable and does not mean CUDA SentenceTransformer inference or LightGBM GPU training failed.

## Result contract

A completed run must produce:

```text
artifacts/eval_summary.json
artifacts/launch_gate.json
artifacts/ranker_bundle.joblib
artifacts/retrieval_bundle.joblib
reports/01_data_validation_report.md
reports/03_lambdarank_training_report.md
reports/04_ablation_report.md
reports/06_slice_reliability_report.md
reports/07_launch_readiness_memo.md
reports/09_claim_boundary.md
```

`PASS`, `REVIEW`, or `ITERATE` are model/data-quality outcomes. They are not execution failures. A runnability failure means an exception, missing required artifacts, non-LambdaRank backend, leakage check failure, or invalid dataset layout.
