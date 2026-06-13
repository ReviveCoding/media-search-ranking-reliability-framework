# Media Search Quality Upgrade Runner v9

This bundle applies and runs the following sequence:

1. Preserve the existing MovieLens 10M baseline and source backup.
2. Apply shared query/catalog normalization.
3. Add similar-to anchor-neighbor candidate generation.
4. Add preference-aware personalized candidate generation.
5. Run unit tests and a synthetic end-to-end regression.
6. Run the core MovieLens 10M GPU quick benchmark.
7. Build optional Tag Genome enrichment and mapping diagnostics.
8. Build optional IMDb exact title/year enrichment and mapping diagnostics.
9. Run the metadata-enriched benchmark when enrichment files are usable.
10. Compare baseline, core-upgrade, and metadata-enriched results.
11. Run API, reproducibility, Monte Carlo, optional full benchmark, and package build.

The runner does not lower launch thresholds. `ITERATE` remains a valid model-quality outcome.
