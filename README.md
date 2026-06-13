# v10.4 Aggregation Contract Hotfix

This patch fixes aggregation when legacy core profile runs were created before the canonical frozen split contract.

- Strict exact contract checks remain required for `core_champion_replay`, `tag_genome_feature_only`, `imdb_feature_only`, and `combined_feature_only`.
- Legacy `core_conservative`, `core_balanced`, and `core_compact` runs remain visible for historical profile-selection context but cannot block aggregation or champion promotion.
- Strict GPU replay is only claimed when manifest, split, and config fingerprint all match.

<!-- FINAL_RESULTS_START -->
## Final frozen benchmark result

- **Promoted champion:** `combined_feature_only`
- **Canonical baseline:** `core_champion_replay`
- **NDCG@10:** `0.312599` → `0.322034` (+3.02%)
- **Frozen contract:** validated for the canonical baseline and enrichment variants
- **Legacy replay disclosure:** `legacy_not_strictly_comparable`; no strict GPU replay claim
- **Launch decision:** `ITERATE`

See [final results](docs/FINAL_RESULTS.md), [reproducibility](docs/REPRODUCIBILITY.md), and [claim boundaries](docs/CLAIM_BOUNDARIES.md).
<!-- FINAL_RESULTS_END -->
