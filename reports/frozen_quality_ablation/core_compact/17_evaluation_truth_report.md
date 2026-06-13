# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |    brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|---------:|
| observed_label |     0.325471 |    0.552902 |       0.195935 |                  0.268735 |          0.186869 |         0.656566 | 0.00497949 | 0.012094 |
| label          |     0.325471 |    0.552902 |       0.195935 |                  0.268735 |          0.186869 |         0.656566 | 0.00497949 | 0.012094 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
