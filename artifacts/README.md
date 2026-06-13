# Artifact policy

Small evaluation summaries and metric tables are intended to be tracked for portfolio review.

The following are generated locally and intentionally ignored because they can be large, environment-specific, or reproducible:

- `ranker_bundle.joblib`
- `retrieval_bundle.joblib`
- `test_predictions.csv`
- `monte_carlo_trials.csv`

Run the demo pipeline to recreate all artifacts:

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml --mode demo
```

Run Monte Carlo validation to recreate stress-test artifacts:

```bash
python scripts/monte_carlo_validate.py --trials-per-scenario 1
```

Run the dataset-only format audit to verify both supported public-data layouts:

```bash
python scripts/dataset_smoke_test.py --quick
```

The compact `dataset_smoke_summary.json` is intended to be tracked.
