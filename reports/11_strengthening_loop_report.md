# Strengthening Loop Report

## Loop 1: Retrieval path credibility

### Weakness
The project built a FAISS-compatible vector index, but candidate generation primarily used direct dense search. This made the FAISS/vector-search claim weaker than the architecture intended.

### Improvement
`HybridRetriever` now accepts an optional `VectorIndex`. When available, dense candidates are retrieved through the vector index, using FAISS if installed and sklearn nearest-neighbor fallback otherwise. The retrieval bundle also stores the vector index backend.

### Verification
- Unit tests: `tests/test_hybrid_retriever.py`
- Demo pipeline: PASS
- Artifact: `artifacts/eval_summary.json` reports `vector_index_backend`.

## Loop 2: Label realism and slice coverage

### Weakness
The first query generator could produce uneven query-type coverage and weak graded-label diversity. This makes NDCG and slice metrics less persuasive.

### Improvement
The query generator now uses stratified query blueprints: `genre_tag`, `similar_to`, `personalized`, `mood_decade`, and `visual_query`. Labels use a full 0/1/2/3 relevance scale and include decade, genre, tag, rating, user preference, and anchor-item signals.

### Verification
- Unit tests: `tests/test_query_labeling_and_slices.py`
- Report: `reports/10_query_label_quality_report.md`
- Demo data now includes all query types and all label grades.

## Loop 3: Quality-latency frontier

### Weakness
The quality-latency report previously used one latency summary for all variants, which weakened the product-readiness analysis.

### Improvement
The pipeline now measures BM25, dense-vector, hybrid, LambdaRank, and calibrated-ranking serving paths separately and joins those latency summaries to the ablation table.

### Verification
- Artifact: `artifacts/latency_by_variant.csv`
- Report: `reports/05_quality_latency_frontier.md`

## Loop 4: Local/GitHub runnability

### Weakness
The repo had tests, but not a single command that validated unit tests, demo pipeline execution, required reports, and non-BLOCK launch readiness.

### Improvement
Added `scripts/smoke_test.py`, `scripts/api_smoke_test.py`, and GitHub Actions CI. The repo now supports fast local verification and GitHub-runnable checks.

### Verification
- `pytest -q`: PASS
- `python scripts/smoke_test.py --mode demo`: PASS
- `python scripts/api_smoke_test.py`: PASS when run after artifacts are generated

## Current demo verification snapshot

- Unit tests: 5 passed
- Launch decision: PASS
- Query types covered: genre, similar-to, personalized, mood/decade, visual-query
- Label grades covered: 0, 1, 2, 3
- Ranker backend: LightGBM LambdaRank CPU in this environment, with GPU fallback support in config
- Dense backend: local fallback in this environment, SentenceTransformer CUDA enabled when installed locally
- Vector index backend: sklearn fallback in this environment, FAISS/FAISS-GPU enabled when installed locally

## Remaining limitations

These are intentional rather than blocking:

1. MovieLens labels are public-data proxy labels, not human search judgments.
2. Demo mode uses synthetic data for fast local validation.
3. GPU acceleration depends on the local CUDA/PyTorch/LightGBM/FAISS installation.
4. Optional MSR-VTT/COCO/CLIP extensions are scaffolded conceptually but not forced into P0 to keep the project compact.
5. The FastAPI app is a local serving demo, not a production deployment claim.

## Final assessment

Further improvements are now minor relative to the project goal. The framework is compact, trainable, end-to-end, public-data bounded, locally runnable, and aligned with media search, ranking, multimodal relevance, and launch-readiness evaluation requirements.

## Loop 5: API contract and serving credibility

### Weakness
The README described an end-to-end serving layer, but the API surface was still too thin for a framework claim. `/batch_search` was documented conceptually but not implemented, `/search` did not expose calibrated scores, and responses did not include review flags or matched relevance signals.

### Improvement
The FastAPI app now supports `/search`, `/batch_search`, `/evaluate`, and `/launch_gate`. Search responses include raw ranking score, calibrated score, matched signals, request latency, and review flags such as `LOW_CONFIDENCE` and `LONG_TAIL_ITEM`. Request validation now bounds query length and `top_k`, and `MEDIA_SEARCH_ARTIFACT_DIR` can override the artifact directory for local/GitHub runs.

### Verification
- Script: `scripts/api_smoke_test.py`
- Result: PASS
- Contract checks: `/health`, `/search`, `/batch_search`, `/launch_gate`, and `/evaluate`

## Loop 6: Environment, GPU-readiness, and runnability audit

### Weakness
The project supported optional GPU use, but there was no explicit preflight report showing whether CUDA, SentenceTransformers, FAISS, and LightGBM were available in the local environment. There was also no single lightweight command that documented local/GitHub runnability without overloading the API/environment checks.

### Improvement
Added `scripts/preflight_check.py` to write `reports/12_environment_and_runnability_report.md` and `artifacts/environment_report.json`. Added `scripts/local_audit.py` to validate one in-process demo pipeline plus repository/artifact contracts. Unit tests, API smoke, reproducibility replay, package build, and Monte Carlo validation run as separate processes to avoid teardown interference in constrained shells.

### Verification
- `python scripts/local_audit.py --mode demo`: PASS
- `python scripts/preflight_check.py`: PASS
- `python scripts/api_smoke_test.py`: PASS
- Reports: `reports/12_environment_and_runnability_report.md`, `reports/13_local_github_runnability_audit.md`

## Final optimization pass

### Cleanup
Removed generated `__pycache__` and `.pyc` files before packaging v3. The repository keeps source code, configs, tests, reports, sample artifacts, and demo data, but avoids Python bytecode noise.

### Assessment
Remaining improvements are now minor relative to the project goal. The project has a trainable ranking model, real retrieval stack, calibrated serving outputs, API/batch API, local/GitHub audit path, explicit GPU-readiness checks, and public/proxy/synthetic claim boundaries.

## Monte Carlo end-to-end validation

- Added configurable synthetic distributions for popularity skew, user preference strength, rating noise, tag sparsity, exploration, cold-start share, and relevance-label noise.
- Corrected recall-denominator and long-tail slice bias.
- Removed similar-to self-match leakage and grounded mood queries.
- Added recall efficiency, stress-direction checks, paired common-random-number scenarios, scenario summaries, and configurable acceptance testing.
- Current lightweight release audit: 4/4 successful paired scenario runs and 20/20 acceptance checks passed. Increase `--trials-per-scenario` for stronger uncertainty estimates.
