# Architecture

```mermaid
flowchart LR
    Q[User query] --> U[Query understanding]
    U --> R[Hybrid retrieval]
    R --> E[Metadata enrichment]
    E --> L[LambdaRank]
    L --> C[Calibration]
    C --> S[Slice evaluation]
    C --> T[Latency evaluation]
    S --> G[Promotion and launch gate]
    T --> G
    F[Frozen manifest and query-group splits] --> R
    F --> L
    F --> S
    G --> A[Release artifacts and claim boundaries]
```

## Reliability layers

### Data and comparison contract

The frozen manifest and query-group split define the canonical comparison boundary. Candidate runs are not treated as directly comparable merely because they use the same dataset family.

### Retrieval and ranking

The framework combines retrieval candidates with feature enrichment and a learning-to-rank stage. Release analysis separates retriever drift from ranker improvement.

### Slice-aware evaluation

Overall ranking quality is evaluated alongside genre/tag, mood/decade, personalization, similar-to, and visual-query slices. Promotion is blocked when material slice regressions exceed the configured guardrail.

### Calibration and latency

Ranking quality is reported with calibration error and latency measurements. Latency remains diagnostic unless measured repeatedly under controlled, identical conditions.

### Promotion and claim governance

A candidate can be the best eligible frozen-contract run while still receiving an `ITERATE` launch decision. Promotion eligibility and production launch readiness are separate decisions.
