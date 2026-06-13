# Evaluation Truth Report

## Training/calibration labels

Observed proxy labels

## Primary evaluation truth

label

## Observed vs primary metrics

| truth_basis    |   ndcg_at_10 |   mrr_at_10 |   recall_at_10 |   recall_efficiency_at_10 |   precision_at_10 |   hit_rate_at_10 |        ece |     brier |
|:---------------|-------------:|------------:|---------------:|--------------------------:|------------------:|-----------------:|-----------:|----------:|
| observed_label |     0.312599 |    0.509456 |       0.196202 |                  0.267388 |          0.187879 |         0.636364 | 0.00490624 | 0.0123183 |
| label          |     0.312599 |    0.509456 |       0.196202 |                  0.267388 |          0.187879 |         0.636364 | 0.00490624 | 0.0123183 |

## Interpretation

Synthetic runs train on noisy observations but evaluate against latent clean relevance. MovieLens runs use proxy observed labels because no latent oracle exists.
