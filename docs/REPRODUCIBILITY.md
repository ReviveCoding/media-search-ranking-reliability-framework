# Reproducibility and Verification

## Supported environment

- Python 3.11
- Windows PowerShell for the repository wrappers
- CPU or CUDA-capable GPU, depending on the selected ranking configuration
- Public MovieLens benchmark inputs and locally generated artifacts

## Fast verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\aggregate_frozen_quality_results.ps1
.\finalize_repository.ps1
```

The aggregation command reuses completed variant summaries. It does not retrain the ranking models.

## Clean-checkout verification

```powershell
.\verify_clean_checkout.ps1
```

This command copies publishable repository files into an isolated temporary directory, creates a fresh virtual environment, installs declared dependencies, runs compilation checks, runs the full test suite, and verifies the frozen release contract.

Use the following command to retain the isolated directory for manual inspection:

```powershell
.\verify_clean_checkout.ps1 -KeepTemp
```

## Frozen comparison contract

The canonical promotion comparison requires exact agreement on the frozen manifest and query-group split. GPU metric replay against older profile-selection runs is diagnostic only because those runs do not share a fully matching configuration fingerprint.

## Expected final state

- Test suite passes
- Frozen aggregation passes
- Canonical contract is validated
- Promoted champion is `combined_feature_only`
- Canonical baseline is `core_champion_replay`
- Legacy strict replay claim remains disabled

## Pinned Python 3.11 dependency contract

The package metadata in `pyproject.toml` defines supported dependency
ranges. Reproducibility and CI additionally use
`constraints/py311.txt`, which pins the dependency set validated with
Python 3.11.9.

Install the validated environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Update the constraint file only after the cross-process reproducibility
replay, full test suite, public portfolio contract, and frozen release
contract all pass.
