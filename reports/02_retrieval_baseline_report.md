# Retrieval Baseline Report

## Data source

C:\Users\bjw-0\Downloads\Project_Data\ml-10m

## Dense backend

sentence-transformers:cuda

## Vector index backend

sklearn-nearest-neighbors-fallback

## Leakage control

Validation/test metrics use retrieval-only candidates; positive-label injection is training-only.

## Metrics

| variant      |   ndcg_at_5 |   mrr_at_5 |   recall_at_5 |   recall_efficiency_at_5 |   precision_at_5 |   hit_rate_at_5 |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |
|:-------------|------------:|-----------:|--------------:|-------------------------:|-----------------:|----------------:|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|
| bm25_score   |   0.0629167 |  0.133333  |    0.0259843  |                0.0533333 |        0.04      |        0.183333 |    0.0504232 |   0.135714  |      0.0280409 |                 0.0407143 |         0.025     |         0.2      |
| dense_score  |   0.030965  |  0.0569444 |    0.00731782 |                0.0266667 |        0.0266667 |        0.116667 |    0.0279949 |   0.0587963 |      0.0125086 |                 0.0216667 |         0.0216667 |         0.133333 |
| hybrid_score |   0.0528725 |  0.0930556 |    0.0242552  |                0.0466667 |        0.0333333 |        0.133333 |    0.0498312 |   0.104048  |      0.0303442 |                 0.0466667 |         0.0316667 |         0.216667 |
