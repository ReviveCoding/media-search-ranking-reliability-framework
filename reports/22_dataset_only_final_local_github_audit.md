# Final Dataset-Only Local and GitHub Audit

## Verdict

**PASS with explicit boundaries.** The framework is runnable when a supported MovieLens-style dataset is placed under `data/raw/movielens/`, and the same dataset-mode path is now exercised in GitHub Actions and from the installed wheel.

## Supported layouts validated

1. MovieLens 1M-style DAT layout:
   - `movies.dat`
   - `ratings.dat`
   - `users.dat`
   - no `tags.dat` required
2. Modern MovieLens CSV layout:
   - `movies.csv`
   - `ratings.csv`
   - optional `tags.csv`

## Validation results

| Contract | Result |
|---|---:|
| Unit/component/contract tests | 37 PASS |
| Synthetic end-to-end local audit | PASS |
| API and batch-search smoke | PASS |
| Cross-process reproducibility | PASS |
| Paired Monte Carlo audit | 20/20 checks PASS |
| DAT dataset-mode end-to-end audit | PASS |
| CSV dataset-mode end-to-end audit | PASS |
| Source and wheel build | PASS |
| Source-independent packaged demo | PASS |
| Source-independent packaged DAT dataset audit | PASS |
| CI YAML parse | PASS |
| Docker runtime build | Not executed in this sandbox |
| Local CUDA/GPU backend | Not available in this sandbox |

## Dataset smoke outputs

| Format | Runnability audit | Launch output | Ranker backend | NDCG@10 | Ranker lift vs hybrid |
|---|---:|---:|---|---:|---:|
| DAT without tags | PASS | REVIEW | `lightgbm-lambdarank-cpu` | 0.4156 | +0.1013 |
| CSV with tags | PASS | REVIEW | `lightgbm-lambdarank-cpu` | 0.3992 | +0.1008 |

`REVIEW` is a model/data-quality decision, not a runtime failure. The dataset-only audit passes when ingestion, feature construction, LambdaRank training, calibration, split-integrity validation, evaluation, and artifact generation complete under the declared backend contracts.

## GitHub enforcement added

The CI workflow now runs:

```text
python scripts/dataset_smoke_test.py --quick
```

The isolated wheel stage also runs:

```text
media-search-dataset-smoke --formats dat --quick
```

This prevents future changes from silently breaking the actual `movielens` execution path while leaving the synthetic demo healthy.

## Source-independent wheel validation

The built wheel was installed outside the repository source tree. Using packaged YAML resources, it completed:

- synthetic demo with `launch_decision=PASS`
- DAT dataset-mode smoke with the full ranking and reporting path
- LightGBM LambdaRank backend verification

## Required local sequence

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/preflight_check.py
python scripts/dataset_smoke_test.py --quick
python scripts/local_audit.py --mode demo
python scripts/api_smoke_test.py
python scripts/reproducibility_check.py
python scripts/monte_carlo_validate.py --trials-per-scenario 1
```

For real MovieLens data:

```powershell
media-search-download --variant 1m
media-search-run --config configs/pipeline.yaml --mode movielens
```

## Boundaries

- The official MovieLens archive could not be downloaded inside this sandbox because outbound DNS was unavailable. The verified fixtures use the exact supported DAT and CSV schemas and execute the same loader and pipeline code.
- MovieLens 1M does not provide tags, so visual/tag query evidence is limited and should not be reported as fully supported multimodal performance.
- Dataset quality can legitimately produce `REVIEW` or `ITERATE`; this does not invalidate runnability.
- Docker was not installed in the sandbox, so Docker execution was not directly verified here. The Dockerfile contract remains covered by tests and GitHub Actions.
- CUDA, SentenceTransformers CUDA, FAISS GPU, and LightGBM GPU were not available in the sandbox. CPU fallback was explicitly reported and validated.
