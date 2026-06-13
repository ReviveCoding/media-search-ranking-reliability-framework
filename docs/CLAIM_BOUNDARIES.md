# Claim Boundaries

## Supported claims

- Built a runnable multimodal media-search and ranking-quality framework with frozen benchmark contracts, ranking evaluation, slice guardrails, calibration, latency reporting, and automated release checks.
- Promoted `combined_feature_only` over the canonical `core_champion_replay` baseline under the same frozen manifest and query-group split.
- Improved NDCG@10 from `0.312599` to `0.322034`, a `+3.02%` relative improvement in the canonical comparison.
- Preserved legacy profile results as historical context while explicitly disabling a strict replay claim when configuration fingerprints do not match.
- Validated repository behavior through automated tests, artifact-contract checks, and an isolated clean-checkout workflow.

## Claims that should not be made

- Do not claim that `core_compact` and `core_champion_replay` are bit-for-bit GPU reproductions.
- Do not describe the legacy score difference as ordinary floating-point noise.
- Do not claim production deployment or online business impact.
- Do not claim that `ITERATE` means launch-ready.
- Do not state a latency improvement as causal unless it is confirmed through repeated, controlled benchmarking on identical hardware and workload.
- Do not imply proprietary user, streaming-platform, or company data.

## Recommended résumé wording

“Built a reproducible media-search retrieval and ranking framework with frozen data/split contracts, LambdaRank evaluation, slice-regression guardrails, calibration and latency diagnostics; promoted a combined metadata feature variant that improved canonical NDCG@10 by 3.02% while preserving explicit GPU replay and launch-readiness boundaries.”

## Frozen benchmark text-encoding limitation

`data/benchmarks/ml10m_frozen_v1/queries.csv` preserves a small
number of legacy movie-title encoding artifacts in anchor-query text.
They are retained in benchmark version `ml10m_frozen_v1` to preserve
the committed comparison contract. Correcting them requires a
versioned benchmark rebuild and recomputation of the frozen results.
