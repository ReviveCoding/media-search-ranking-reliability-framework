# Frozen MovieLens Ranking Quality Upgrade

This upgrade improves real MovieLens ranking experiments without changing the evaluation problem between runs.

## What changes

1. Freezes query text, proxy judgments, user context, and train/validation/test query groups.
2. Adds intent-aware candidate quotas for genre/tag, mood/decade, personalized, similar-to, and visual queries.
3. Reports candidate Recall Efficiency@50/@100 by source and query type.
4. Uses Tag Genome and IMDb in feature-only mode by default, so metadata does not rewrite retrieval documents or labels.
5. Adds numeric Tag Genome and IMDb ranking features.
6. Runs a small, leakage-controlled LightGBM LambdaRank profile search with early stopping.
7. Promotes metadata variants only when they do not regress any query-type NDCG by more than 10% and do not change retrieval text.

## Quick run

```powershell
cd "C:\Users\bjw-0\Downloads\media-search-multimodal-discovery-reliability-framework"
Set-ExecutionPolicy -Scope Process Bypass -Force
Unblock-File ".\run_movieLens_frozen_quality_ablation.ps1"
.\run_movieLens_frozen_quality_ablation.ps1
```

## Full run

```powershell
.\run_movieLens_frozen_quality_ablation.ps1 -RunFull
```

## Faster core-only diagnostic

```powershell
.\run_movieLens_frozen_quality_ablation.ps1 -SkipRankerTuning
```

## Main outputs

- `data/benchmarks/ml10m_frozen_v1/manifest.json`
- `data/benchmarks/ml10m_frozen_v1/queries.csv`
- `data/benchmarks/ml10m_frozen_v1/judgments.csv`
- `artifacts/frozen_quality_ablation/frozen_ablation_results.csv`
- `artifacts/frozen_quality_ablation/frozen_ablation_summary.json`
- `reports/frozen_quality_ablation/29_frozen_movielens_ablation.md`
- `<variant artifact dir>/candidate_recall_diagnostics.csv`

## Interpretation

- Low candidate recall means candidate generation is the bottleneck.
- High candidate recall with low NDCG means reranking is the bottleneck.
- A metadata variant is not promoted when it changes retrieval text, loses ranker lift, reduces overall NDCG, or regresses a query-type NDCG by more than 10%.
