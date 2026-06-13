# Split Integrity Report

## Strategy

frozen-benchmark-manifest

## Split sizes

{
  "train": 324,
  "val": 77,
  "test": 99
}

## Anchor overlap

{}

## Personalized-user overlap

{}

## Query-type counts

{
  "train": {
    "visual_query": 66,
    "genre_tag": 65,
    "personalized": 65,
    "mood_decade": 65,
    "similar_to": 63
  },
  "val": {
    "genre_tag": 16,
    "similar_to": 16,
    "personalized": 15,
    "mood_decade": 15,
    "visual_query": 15
  },
  "test": {
    "similar_to": 21,
    "personalized": 20,
    "mood_decade": 20,
    "visual_query": 19,
    "genre_tag": 19
  }
}

## Interpretation

Anchor-linked queries and repeated personalized users are assigned to one split to reduce train/validation/test leakage.
