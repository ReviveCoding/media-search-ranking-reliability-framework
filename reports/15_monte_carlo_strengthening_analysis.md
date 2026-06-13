# Monte Carlo Strengthening and Remediation Analysis

## Final decision

**PASS**: 4/4 paired scenario runs completed and 20/20 acceptance checks passed.

## Problems found and resolved

| Problem found | Why it mattered | Remediation | Final evidence |
|---|---|---|---|
| Optimistic recall denominator | Candidate-only denominators can hide relevant items missed by retrieval. | Recall and ideal DCG use an external judgment set; recall efficiency accounts for top-k capacity. | Minimum Recall Efficiency@10 was 0.600. |
| Long-tail and cold-start re-ranking bias | Filtering a slice before ranking overstates discovery. | Slice metrics are measured inside the full ranked top-k. Hit rate is used for the launch gate; raw recall and efficiency remain diagnostics. | Final demo long-tail/cold-start hit rates were 0.667/0.733. |
| Similar-to self-match | Returning the anchor item inflates MRR without recommending an alternative. | Anchor labels are zero, the anchor is excluded before ranking and serving, and a regression test covers the behavior. | Maximum self-return@10 was 0.000. |
| Noisy truth contamination | If the same noisy label drives training and evaluation, stress degradation is hidden. | Synthetic mode separates latent clean relevance from observed noisy training labels. | All successful trials evaluated against `clean_label`. |
| Train-serving skew | Offline structured query features were initially missing at API inference. | Shared query-understanding code reconstructs genre, tag, mood, decade, anchor, and user-context features during serving. | API smoke test covers search and batch search. |
| Split leakage | Similar-to anchors and repeated personalized users could cross train/validation/test. | Group-aware splitting keeps linked anchors and personalized users in one split. | Anchor and personalized-user leakage checks passed in every trial. |
| Scenario confounding | Independent scenario seeds mix stress effects with catalog/user randomness. | Paired common-random-number trials reuse the same base seed across scenarios. | High-noise and sparse-metadata direction checks passed. |
| Weak visual-stress diagnostic | One held-out visual query can dominate a small test split. | Sparse-metadata validation uses retrieval-only diagnostics over all generated visual queries. | Sparse metadata reduced paired visual-query quality. |
| Cross-process nondeterminism | Hash/set ordering and tied scores could change outputs across runs. | Stable sorting, deterministic LightGBM seeds, single-thread ranker training, and tie-breaking by `movie_id` were added. | Cross-process reproducibility passed at 1e-12 prediction tolerance. |
| Repeated-run resource instability | Native numerical runtimes can over-create threads. | Monte Carlo uses bounded threads, small-data profiles, process isolation where appropriate, and garbage collection. | All paired stress runs completed. |

## Final observed range

| Metric | Result |
|---|---:|
| NDCG@10 | 0.610 to 0.774 |
| Median ranker lift vs hybrid | +0.150 |
| Minimum Recall Efficiency@10 | 0.600 |
| Maximum ECE | 0.173 |
| Maximum p95 local latency | 44.0 ms |
| Maximum similar-to self-return@10 | 0.000 |

## Remaining limitations

- The default validation uses one paired trial per scenario for fast regression coverage. It is not a high-confidence uncertainty study.
- Synthetic labels and MovieLens-derived labels are offline proxies, not human search judgments.
- The default multimodal path uses visual-intent queries and tag/scene metadata as a proxy. CLIP, video encoders, and VLMs are not claimed by the default run.
- This environment validated CPU fallbacks only. CUDA SentenceTransformers, FAISS GPU, and LightGBM GPU require the target local device.
- Local latency is a small-data engineering diagnostic, not production-scale latency evidence.

## Conclusion

The remaining improvements are external validation, larger Monte Carlo trial counts, optional real multimodal features, and target-device GPU benchmarks. These are evidence-expansion tasks rather than framework-correctness blockers.
