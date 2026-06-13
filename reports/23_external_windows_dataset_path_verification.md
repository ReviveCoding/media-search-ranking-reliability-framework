# External Windows Dataset Path Verification

## Scope

The framework was hardened to read MovieLens data directly from an external directory instead of requiring a copy under `data/raw/movielens`.

Supported precedence:

1. `MOVIELENS_RAW_DIR` environment variable.
2. `data.movielens_raw_dir` in YAML.
3. Repository-local `data/raw/movielens` fallback.

The path resolver accepts direct and nested archive layouts containing either:

- `movies.dat` and `ratings.dat`, with optional `tags.dat`, or
- `movies.csv` and `ratings.csv`, with optional `tags.csv`.

## Verification performed

A small nested MovieLens CSV fixture was written below a parent directory, while the pipeline was given only the parent path. The resolver selected the nested dataset directory and the complete `movielens` pipeline completed.

```text
Path resolution: PASS
Dataset ingestion: PASS
Hybrid retrieval: PASS
LightGBM LambdaRank: PASS
Calibration: PASS
Held-out evaluation: PASS
Report and artifact generation: PASS
```

Observed small-fixture output:

```text
Launch output: REVIEW
Ranker backend: lightgbm-lambdarank-cpu
Dense backend: tfidf-normalized-fallback
Vector backend: sklearn-nearest-neighbors-fallback
NDCG@10: 0.5667
Ranker NDCG lift vs hybrid: +0.1464
```

`REVIEW` is a model/data quality outcome on the small fixture, not a runnability failure.

## Windows helper

Use `scripts/run_windows_local.ps1` to select MovieLens 10M, 20M, or 32M without moving the source files. The script validates the path, runs the environment preflight, optionally runs tests, and starts the GPU-oriented MovieLens pipeline.

## Validation boundary

The supplied Windows directories cannot be mounted inside this sandbox. Therefore, their actual file contents and the local CUDA backend must be confirmed on the user's machine using:

```powershell
media-search-check-path "C:\Users\bjw-0\Downloads\Project_Data\ml-10m"
python scripts/preflight_check.py
```
