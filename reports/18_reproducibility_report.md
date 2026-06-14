# Cross-Process Reproducibility Report

**Decision:** PASS

The same compact CPU-safe pipeline was executed in two independent Python processes with different `PYTHONHASHSEED` values (1 and 999).

Verified identical outputs:

- synthetic queries
- clean and observed graded relevance labels
- split diagnostics
- model-quality metrics excluding wall-clock latency
- row-level test predictions within absolute tolerance `1e-12`

Reference NDCG@10: `0.721586`
Reference ranker lift: `+0.097400`

Latency is intentionally excluded because wall-clock measurements are environment dependent.