# Quality Upgrade Comparison

| variant | ndcg@10 | recall_efficiency@10 | mrr@10 | similar_to_ndcg@10 | personalized_ndcg@10 | mood_decade_ndcg@10 | genre_tag_ndcg@10 | ranker_lift | launch | dense_backend | ranker_backend |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1543196251056228 | 0.12904761904761902 | 0.3518452380952381 | 0.08092925023403873 | 0.09322214741064412 | 0.03497523037022456 | 0.2599501326735815 | 0.10448840326059433 | ITERATE | sentence-transformers:cuda | lightgbm-lambdarank-gpu |
| core_upgrade | 0.24114339299325663 | 0.1850952380952381 | 0.5 | 0.1810461101648673 | 0.18900129145123756 | 0.029770178643897683 | 0.3659806576583933 | 0.2104841906045213 | ITERATE | sentence-transformers:cuda | lightgbm-lambdarank-gpu |
| metadata_enriched | 0.1694249037836338 | 0.1259259259259259 | 0.3645833333333333 | 0.0749989395602315 | 0.07390201246131102 | 0.037613594888352406 | 0.2635140804658178 | 0.1514201539365214 | ITERATE | sentence-transformers:cuda | lightgbm-lambdarank-gpu |

## Interpretation

The core upgrade should improve similar-to and personalized slices without regressing genre/tag search materially. Metadata enrichment should only be claimed when mapping coverage is documented in reports 24–26.
