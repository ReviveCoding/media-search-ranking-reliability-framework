# Data directory

Raw MovieLens downloads and generated processed/synthetic tables are intentionally excluded from source control.

Generate a synthetic demo locally:

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml --mode demo
```

Download and run MovieLens:

```bash
python scripts/download_movielens.py --variant 1m
python scripts/run_pipeline.py --config configs/pipeline.yaml --mode movielens
```

See `DATA_SOURCES_AND_LICENSES.md` before redistributing any public dataset files.
