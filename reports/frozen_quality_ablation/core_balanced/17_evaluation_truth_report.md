# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |     0.307701 |    0.517448 |       0.191191 |                  0.258297 |          0.178788 |         0.626263 | 0.00478691 | 0.0124271 |
| label          |     0.307701 |    0.517448 |       0.191191 |                  0.258297 |          0.178788 |         0.626263 | 0.00478691 | 0.0124271 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
