# Quality-Latency Frontier

## Per-variant latency

| variant      |   p50_latency_ms |   p95_latency_ms |   p99_latency_ms |
|:-------------|-----------------:|-----------------:|-----------------:|
| bm25_score   |          34.3577 |          40.2913 |          42.6377 |
| dense_score  |          14.8001 |          16.457  |          16.9824 |
| hybrid_score |          48.1857 |          58.1369 |          62.5209 |
| ranker_score |         130.98   |         143.108  |         144.581  |

## Quality-latency table

| variant      |   ndcg_at_5 |   mrr_at_5 |   recall_at_5 |   recall_efficiency_at_5 |   precision_at_5 |   hit_rate_at_5 |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |   p50_latency_ms |   p95_latency_ms |   p99_latency_ms |
|:-------------|------------:|-----------:|--------------:|-------------------------:|-----------------:|----------------:|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| bm25_score   |   0.0629167 |  0.133333  |    0.0259843  |                0.0533333 |        0.04      |        0.183333 |    0.0504232 |   0.135714  |      0.0280409 |                 0.0407143 |         0.025     |         0.2      |          34.3577 |          40.2913 |          42.6377 |
| dense_score  |   0.030965  |  0.0569444 |    0.00731782 |                0.0266667 |        0.0266667 |        0.116667 |    0.0279949 |   0.0587963 |      0.0125086 |                 0.0216667 |         0.0216667 |         0.133333 |          14.8001 |          16.457  |          16.9824 |
| hybrid_score |   0.0528725 |  0.0930556 |    0.0242552  |                0.0466667 |        0.0333333 |        0.133333 |    0.0498312 |   0.104048  |      0.0303442 |                 0.0466667 |         0.0316667 |         0.216667 |          48.1857 |          58.1369 |          62.5209 |
| ranker_score |   0.195312  |  0.373889  |    0.045651   |                0.16      |        0.146667  |        0.466667 |    0.165901  |   0.388003  |      0.0627936 |                 0.137381  |         0.121667  |         0.583333 |         130.98   |         143.108  |         144.581  |

## Interpretation

The expensive learned ranker operates only after candidate retrieval. Calibrated score is reported for confidence and launch gating rather than as a separate retrieval model.
