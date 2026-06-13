# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |     0.302419 |    0.484315 |       0.192206 |                   0.25942 |          0.179798 |         0.626263 | 0.00385446 | 0.0127993 |
| label          |     0.302419 |    0.484315 |       0.192206 |                   0.25942 |          0.179798 |         0.626263 | 0.00385446 | 0.0127993 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
