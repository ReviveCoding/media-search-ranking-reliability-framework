# Public Release Checklist

## Repository content

- [ ] README opens with architecture and results at a glance.
- [ ] CI badge resolves after the first Actions run.
- [ ] `python scripts/public_demo.py` runs.
- [ ] Full tests pass.
- [ ] `verify_clean_checkout.ps1` passes against committed `HEAD`.
- [ ] No secrets, caches, local paths, or large runtime artifacts are tracked.
- [ ] Claim boundaries remain explicit.

## Suggested GitHub description

```text
Reproducible media-search framework for hybrid retrieval, LambdaRank reranking, personalization, slice-aware evaluation, calibration, latency, and frozen release contracts.
```

## Suggested topics

```text
machine-learning
information-retrieval
learning-to-rank
search-ranking
recommendation-systems
multimodal-search
model-evaluation
ml-reliability
reproducibility
lightgbm
python
```

## Social preview

Upload `docs/assets/social-preview.png` through:

```text
Repository â†’ Settings â†’ Social preview â†’ Edit â†’ Upload an image
```

## GitHub settings

- Enable Dependabot alerts.
- Enable secret scanning.
- Enable push protection.
- Enable code scanning when practical.
- Confirm both Actions workflows are green.
- Pin the repository on the profile.
