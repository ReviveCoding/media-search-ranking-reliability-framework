# Public Demo Walkthrough

## Release summary demo

```powershell
.\\.venv\\Scripts\\python.exe scripts\\public_demo.py
```

The command reads committed benchmark and release artifacts and prints one frozen benchmark query, the system flow, canonical metric change, contract state, replay-claim state, and launch decision.

## Full regression suite

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```

Expected after this polish:

```text
61 passed
```

## Contract checks

```powershell
.\\.venv\\Scripts\\python.exe scripts\\verify_release_contract.py --repo .
.\\.venv\\Scripts\\python.exe scripts\\verify_public_portfolio.py --repo .
```

## Exact committed snapshot

```powershell
.\\verify_clean_checkout.ps1
```

This uses `git archive HEAD`, installs the committed snapshot into a new virtual environment, runs compilation and tests, and verifies the frozen release contract.
