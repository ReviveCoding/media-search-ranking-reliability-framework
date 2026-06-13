# Launch Readiness Memo

## Decision

ITERATE

## Checks

| check                                  | passed   | value                      | threshold                    |
|:---------------------------------------|:---------|:---------------------------|:-----------------------------|
| ndcg_at_10                             | True     | 0.3254711389373969         | 0.3                          |
| recall_efficiency_at_10                | False    | 0.2687349687349687         | 0.5                          |
| mrr_at_10                              | True     | 0.5529020362353696         | 0.2                          |
| ece                                    | True     | 0.004979491359308477       | 0.18                         |
| p95_latency_ms                         | True     | 179.1937999980291          | 500                          |
| long_tail_hit_rate                     | False    | 0.04225352112676056        | 0.5                          |
| cold_start_hit_rate                    | False    | 0.037037037037037035       | 0.5                          |
| ranker_ndcg_lift                       | True     | 0.2079483072248821         | 0.01                         |
| similar_to_self_return                 | True     | 0.0                        | 0.0                          |
| ranker_backend_contract                | True     | lightgbm-lambdarank-gpu    | prefix:lightgbm-lambdarank   |
| dense_backend_contract                 | True     | sentence-transformers:cuda | prefix:sentence-transformers |
| approved_public_or_synthetic_data_only | True     | True                       | True                         |
| no_private_user_data                   | True     | True                       | True                         |
| no_production_deployment_claim         | True     | True                       | True                         |

## Recommendation

PASS means configured offline thresholds were met. Monte Carlo stability is assessed separately and should be reviewed before resume claims.
