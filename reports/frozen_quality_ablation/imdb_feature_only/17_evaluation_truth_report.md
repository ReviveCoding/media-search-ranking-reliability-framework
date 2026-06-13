# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |     0.304503 |     0.50683 |       0.193115 |                  0.262225 |          0.182828 |         0.636364 | 0.00446411 | 0.0123732 |
| label          |     0.304503 |     0.50683 |       0.193115 |                  0.262225 |          0.182828 |         0.636364 | 0.00446411 | 0.0123732 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
