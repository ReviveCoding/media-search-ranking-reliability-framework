# Query and Label Quality Report

## Judgment coverage

Pooled proxy judgments for memory-efficient MovieLens mode.

## Query-type coverage

| query_type   |   count |
|:-------------|--------:|
| genre_tag    |      60 |
| similar_to   |      60 |
| personalized |      60 |
| mood_decade  |      60 |
| visual_query |      60 |

## Graded label distribution

|   label |   count |
|--------:|--------:|
|       0 |    9572 |
|       1 |    5155 |
|       2 |    4058 |
|       3 |    5215 |

## Positive-label coverage

| index   |   query_id |   max_label |   positive_count |
|:--------|-----------:|------------:|-----------------:|
| count   |   300      |  300        |         300      |
| mean    |   149.5    |    2.93667  |          30.91   |
| std     |    86.7468 |    0.257312 |          12.9031 |
| min     |     0      |    1        |           0      |
| 25%     |    74.75   |    3        |          21.75   |
| 50%     |   149.5    |    3        |          31      |
| 75%     |   224.25   |    3        |          40      |
| max     |   299      |    3        |          65      |

## Interpretation

Labels remain public/synthetic proxy judgments rather than human search judgments. Retrieval metrics use the external judgment set as the denominator.
