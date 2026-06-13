# Split Integrity Report

## Strategy

group-aware-best-of-random-candidates

## Split sizes

{
  "train": 324,
  "val": 77,
  "test": 99
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
    "genre_tag": 65,
    "mood_decade": 65,
    "personalized": 65,
    "similar_to": 63,
    "visual_query": 66
  },
  "val": {
    "genre_tag": 16,
    "mood_decade": 15,
    "personalized": 15,
    "similar_to": 16,
    "visual_query": 15
  },
  "test": {
    "genre_tag": 19,
    "mood_decade": 20,
    "personalized": 20,
    "similar_to": 21,
    "visual_query": 19
  }
}

## Interpretation

Anchor-linked queries and repeated personalized users are assigned to one split to reduce train/validation/test leakage.
