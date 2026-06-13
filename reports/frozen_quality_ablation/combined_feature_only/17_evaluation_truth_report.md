# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |     0.322034 |    0.546196 |       0.195613 |                  0.265705 |          0.183838 |         0.666667 | 0.00517651 | 0.0121426 |
| label          |     0.322034 |    0.546196 |       0.195613 |                  0.265705 |          0.183838 |         0.666667 | 0.00517651 | 0.0121426 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
