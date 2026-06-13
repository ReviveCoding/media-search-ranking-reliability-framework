# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |      0.30831 |     0.51158 |       0.193293 |                  0.263236 |          0.183838 |         0.636364 | 0.00371676 | 0.0124277 |
| label          |      0.30831 |     0.51158 |       0.193293 |                  0.263236 |          0.183838 |         0.636364 | 0.00371676 | 0.0124277 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
