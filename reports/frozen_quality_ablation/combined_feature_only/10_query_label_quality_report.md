# Query and Label Quality Report

## Judgment coverage

Pooled proxy judgments for memory-efficient MovieLens mode.

## Frozen benchmark

{
  "version": 1,
  "seed": 42,
  "data_source": "C:\\Users\\bjw-0\\Downloads\\Project_Data\\ml-10m",
  "catalog_fingerprint": "82c8f9cbf0375e41e2a1f1eafcad5dac99e5b5d33d005e34d3de5c7d8e2a196b",
  "catalog_size": 10000,
  "query_count": 500,
  "judgment_count": 50000
}

## Query-type coverage

| query_type   |   count |
|:-------------|--------:|
| genre_tag    |     100 |
| similar_to   |     100 |
| personalized |     100 |
| mood_decade  |     100 |
| visual_query |     100 |

## Graded label distribution

|   label |   count |
|--------:|--------:|
|       0 |   27634 |
|       1 |    9298 |
|       2 |    9582 |
|       3 |    3486 |

## Positive-label coverage

| index   |   query_id |   max_label |   positive_count |
|:--------|-----------:|------------:|-----------------:|
| count   |    500     |  500        |         500      |
| mean    |    249.5   |    2.692    |          26.136  |
| std     |    144.482 |    0.466445 |          17.8609 |
| min     |      0     |    1        |           0      |
| 25%     |    124.75  |    2        |           9      |
| 50%     |    249.5   |    3        |          26      |
| 75%     |    374.25  |    3        |          38      |
| max     |    499     |    3        |          72      |

## Interpretation

Labels remain public/synthetic proxy judgments rather than human search judgments. Frozen ablations reuse identical queries, judgments, and splits.
