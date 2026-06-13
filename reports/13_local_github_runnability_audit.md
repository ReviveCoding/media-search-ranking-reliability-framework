# Local and GitHub Runnability Audit

**Decision:** PASS

This audit intentionally runs one end-to-end pipeline in-process and checks the repository/artifact contract. Unit tests, API smoke, reproducibility replay, package build, and Monte Carlo validation run as separate processes in CI to avoid OpenMP/TestClient teardown interference in constrained shells.

| Check | Status | Detail |
|---|---|---|
| installed_package_import | PASS | media_search_reliability 0.6.0 |
| source_compilation | PASS | src/ and scripts/ compile |
| repository_contract | PASS | missing=[] |
| end_to_end_pipeline | PASS | launch=PASS |
| split_integrity | PASS | {"anchor_leakage_free": true, "personalized_user_leakage_free": true} |
| artifact_contract | PASS | missing=[] |

## Separate full-validation commands

```bash
python -m pytest -q
python scripts/local_audit.py --mode demo
python scripts/api_smoke_test.py
python scripts/reproducibility_check.py
python scripts/monte_carlo_validate.py --trials-per-scenario 1
python -m build
```
