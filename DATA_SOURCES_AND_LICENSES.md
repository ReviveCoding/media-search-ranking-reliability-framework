# Data Sources and Licenses

This repository does not redistribute MovieLens, MSR-VTT, COCO, Apple data, or private user data.

## Core public dataset

- MovieLens 1M / 10M: download from the official GroupLens dataset pages or with `scripts/download_movielens.py`.
- Respect the license and usage terms distributed with the selected MovieLens archive.

## Synthetic validation data

The demo and Monte Carlo datasets are generated locally by this repository. They contain simulated movie metadata, user preferences, interactions, ratings, tags, query intents, and relevance labels. They do not represent real people or real Apple product usage.

## Optional extensions

MSR-VTT and COCO are mentioned only as optional external extensions. They are not downloaded or included by default. Review their current official terms before use.

## Claim boundary

See `reports/09_claim_boundary.md`. Public and synthetic data results must not be described as Apple-internal, production, or real-user results.
