# Model Card

## Model purpose

Public-data media search and discovery ranking for offline evaluation and local serving demos.

## Multimodal scope

The default path uses visual-intent queries and scene/tag metadata as a lightweight cross-modal proxy. It does not claim CLIP, video-encoder, or VLM training unless an optional extension is explicitly enabled and separately reported.

## Training data

MovieLens-style public ratings/tags plus synthetic query labels and synthetic user-context features.

## Retrieval model

BM25, dense retrieval, vector index, and hybrid fusion. Dense backend: sentence-transformers:cuda.

## Ranking model

lightgbm-lambdarank-gpu with query-grouped graded relevance labels.

## Known limitations

Synthetic query labels are proxies for human search relevance. MovieLens labels are also offline proxies. GPU backends, optional encoders, and production-scale latency require separate validation in the target environment. No real Apple, Siri, Spotlight, Apple TV, or private user data is used.

## Launch decision

ITERATE
