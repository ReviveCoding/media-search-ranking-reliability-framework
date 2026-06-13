# Quality-Latency Frontier

## Per-variant latency

| variant      |   p50_latency_ms |   p95_latency_ms |   p99_latency_ms |
|:-------------|-----------------:|-----------------:|-----------------:|
| bm25_score   |          26.6956 |          39.009  |          47.3661 |
| dense_score  |          27.0801 |          29.5265 |          30.3458 |
| hybrid_score |          56.3996 |          76.0289 |          87.2452 |
| ranker_score |         167.277  |         210.466  |         291.834  |

## Quality-latency table

| variant      |   ndcg_at_5 |   mrr_at_5 |   recall_at_5 |   recall_efficiency_at_5 |   precision_at_5 |   hit_rate_at_5 |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |   p50_latency_ms |   p95_latency_ms |   p99_latency_ms |
|:-------------|------------:|-----------:|--------------:|-------------------------:|-----------------:|----------------:|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| bm25_score   |   0.111022  |   0.170034 |     0.0688341 |                0.102189  |        0.0787879 |        0.212121 |    0.103026  |    0.175613 |      0.0843406 |                 0.0977674 |         0.0575758 |         0.252525 |          26.6956 |          39.009  |          47.3661 |
| dense_score  |   0.0915162 |   0.14798  |     0.0543667 |                0.0840067 |        0.0646465 |        0.191919 |    0.0875129 |    0.153171 |      0.071255  |                 0.0832251 |         0.0515152 |         0.232323 |          27.0801 |          29.5265 |          30.3458 |
| hybrid_score |   0.124342  |   0.20404  |     0.0720418 |                0.111448  |        0.0888889 |        0.262626 |    0.117523  |    0.207728 |      0.0950706 |                 0.110382  |         0.0666667 |         0.292929 |          56.3996 |          76.0289 |          87.2452 |
| ranker_score |   0.361601  |   0.480808 |     0.173347  |                0.338552  |        0.292929  |        0.59596  |    0.302419  |    0.484315 |      0.192206  |                 0.25942   |         0.179798  |         0.626263 |         167.277  |         210.466  |         291.834  |

## Interpretation

The expensive learned ranker operates only after candidate retrieval. Calibrated score is reported for confidence and launch gating rather than as a separate retrieval model.
