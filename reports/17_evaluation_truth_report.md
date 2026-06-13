# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |     0.165901 |    0.388003 |      0.0627936 |                  0.137381 |          0.121667 |         0.583333 | 0.00373044 | 0.0129769 |
| label          |     0.165901 |    0.388003 |      0.0627936 |                  0.137381 |          0.121667 |         0.583333 | 0.00373044 | 0.0129769 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
