# Release Hardening and Final Audit Report

## Decision

**PASS with explicit external-validation boundaries.**

This loop re-opened the previous optimized release and found additional issues that mattered for installability, backend integrity, Windows/GPU portability, serving latency interpretation, and small-slice claim safety. Each material issue was corrected and revalidated on compact synthetic data.

## Weakness-resolution loops

| Loop | Material weakness | Resolution | Verification |
|---|---|---|---|
| 1 | Tests imported `scripts.*`, which is not part of the installed package | Moved reusable download and Monte Carlo sampling logic into `src/media_search_reliability` | Source-isolated tests pass with `PYTHONPATH=src` |
| 2 | LambdaRank failures could silently become a generic regressor | Default now refuses non-ranking fallback; GPU may retry CPU LambdaRank; explicit diagnostic fallback is labeled | Backend contract tests and launch-gate check pass |
| 3 | Explicit SentenceTransformers requests could silently become TF-IDF | Added strict/permissive dense-backend modes and backend launch contracts | Dense backend contract tests pass |
| 4 | `gpu` extra bundled FAISS in a Windows-unfriendly way | Split transformer GPU and optional FAISS extras; retained labeled vector-index fallback | `pyproject.toml` parse/build and portability tests pass |
| 5 | Installed wheel had no default YAML configuration | Packaged default configs and added repo/package drift test | Wheel runs end to end outside the source checkout |
| 6 | CLI help eagerly imported the ML stack | Deferred heavy imports until after argument parsing | CLI import-boundary test and wheel CLI help pass |
| 7 | Docker packaging omitted `LICENSE` before `pip install .` | Corrected Docker metadata copy order and added static contract test | Docker contract test passes; runtime Docker unavailable here |
| 8 | API mixed one-time model loading with steady-state query latency | `/ready` now preloads under a lock; responses separate load, search, and total latency | Wheel API test shows cold load and warm search separately |
| 9 | Query-slice scores could look definitive with only a few held-out queries | Added `num_queries` and `claim_support`; small slices are `LOW_SUPPORT` | Slice support test and generated report pass |
| 10 | `make clean` removed hiring-manager-readable evidence | `clean` preserves reports and compact artifacts; `clean-all` is explicit | Makefile contract and release cleanup verified |

## Final compact-demo metrics

- Launch decision: **PASS**
- Ranker backend: `lightgbm-lambdarank-cpu`
- Dense backend in this sandbox: `tfidf-normalized-fallback`
- Vector index in this sandbox: `sklearn-nearest-neighbors-fallback`
- NDCG@10: `0.690036`
- MRR@10: `0.966667`
- Recall efficiency@10: `0.706667`
- Ranker lift vs hybrid NDCG@10: `+0.183884`
- ECE: `0.100659`
- Pipeline steady-state p95 latency: `49.812 ms`
- Similar-to self-return@10: `0.000000`

These are synthetic CPU-fallback smoke metrics, not MovieLens, human-judged, GPU, or production results.

## Slice claim support

The demo intentionally keeps data small. Query-type test slices below the configured support threshold are not standalone benchmark evidence.

| slice                   |   num_queries | claim_support   |
|:------------------------|--------------:|:----------------|
| query_type:genre_tag    |             3 | LOW_SUPPORT     |
| query_type:mood_decade  |             3 | LOW_SUPPORT     |
| query_type:personalized |             3 | LOW_SUPPORT     |
| query_type:similar_to   |             3 | LOW_SUPPORT     |
| query_type:visual_query |             3 | LOW_SUPPORT     |

## Monte Carlo regression audit

- Decision: **PASS**
- Successful trials: `4/4`
- Acceptance checks: `20/20`
- Scenarios: nominal, high noise, long-tail skew, sparse metadata

## Final runnability contract

Verified in this environment:

- 35 unit/component/contract tests
- source-isolated import and test execution
- end-to-end synthetic pipeline and launch gate
- FastAPI health/readiness/search/batch paths
- cross-process deterministic replay
- paired Monte Carlo directional stress audit
- isolated sdist and wheel build
- installed wheel E2E execution from a directory without the repository, using an already validated runtime dependency set
- packaged default configuration fallback
- three installed CLI entry points
- safe download/extraction checks

Not directly verified here:

- CUDA/SentenceTransformers inference on the user's device
- a platform-compatible FAISS GPU build
- LightGBM GPU build support
- Docker image build, because Docker is unavailable in this sandbox
- full dependency resolution in a brand-new empty virtual environment, because the sandbox package index did not complete within the validation window
- MovieLens 1M/10M final benchmark
- human relevance judgments

## Stop criterion

No remaining issue changes framework correctness, leakage control, package installability, CPU-safe E2E execution, or the honesty of current claims. Additional work now has lower marginal value and should focus on external evidence: local GPU execution, MovieLens benchmarking, more Monte Carlo replications, and optional real multimodal features.
