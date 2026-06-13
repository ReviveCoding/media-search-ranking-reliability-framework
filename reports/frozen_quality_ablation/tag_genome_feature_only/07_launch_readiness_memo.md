# Launch Readiness Memo

## Decision

ITERATE

## Checks

| check                                  | passed   | value                      | threshold                    |
|:---------------------------------------|:---------|:---------------------------|:-----------------------------|
| ndcg_at_10                             | True     | 0.30241949288809516        | 0.3                          |
| recall_efficiency_at_10                | False    | 0.2594195927529261         | 0.5                          |
| mrr_at_10                              | True     | 0.48431537598204266        | 0.2                          |
| ece                                    | True     | 0.003854464203776614       | 0.18                         |
| p95_latency_ms                         | True     | 210.46639000051073         | 500                          |
| long_tail_hit_rate                     | False    | 0.014084507042253521       | 0.5                          |
| cold_start_hit_rate                    | False    | 0.018518518518518517       | 0.5                          |
| ranker_ndcg_lift                       | True     | 0.1848966611755804         | 0.01                         |
| similar_to_self_return                 | True     | 0.0                        | 0.0                          |
| ranker_backend_contract                | True     | lightgbm-lambdarank-gpu    | prefix:lightgbm-lambdarank   |
| dense_backend_contract                 | True     | sentence-transformers:cuda | prefix:sentence-transformers |
| approved_public_or_synthetic_data_only | True     | True                       | True                         |
| no_private_user_data                   | True     | True                       | True                         |
| no_production_deployment_claim         | True     | True                       | True                         |

## Recommendation

PASS means configured offline thresholds were met. Monte Carlo stability is assessed separately and should be reviewed before resume claims.
