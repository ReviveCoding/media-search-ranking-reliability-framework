from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from media_search_reliability.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--mode", choices=["demo", "movielens"], default="demo")
    args = parser.parse_args()
    summary = run_pipeline(args.config, mode=args.mode)
    print("\nPipeline complete.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
