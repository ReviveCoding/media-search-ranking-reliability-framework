# Split Integrity Report

## Strategy

group-aware-best-of-random-candidates

## Split sizes

{
  "train": 193,
  "val": 47,
  "test": 60
}

## Anchor overlap

{
  "train_val": 0,
  "train_test": 0,
  "val_test": 0
}

## Personalized-user overlap

{
  "train_val": 0,
  "train_test": 0,
  "val_test": 0
}

## Query-type counts

{
  "train": {
    "genre_tag": 41,
    "mood_decade": 37,
    "personalized": 38,
    "similar_to": 39,
    "visual_query": 38
  },
  "val": {
    "genre_tag": 9,
    "mood_decade": 9,
    "personalized": 10,
    "similar_to": 9,
    "visual_query": 10
  },
  "test": {
    "genre_tag": 10,
    "mood_decade": 14,
    "personalized": 12,
    "similar_to": 12,
    "visual_query": 12
  }
}

## Interpretation

Anchor-linked queries and repeated personalized users are assigned to one split to reduce train/validation/test leakage.
