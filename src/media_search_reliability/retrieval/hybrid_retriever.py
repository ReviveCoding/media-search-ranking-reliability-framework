from __future__ import annotations

import numpy as np
from media_search_reliability.utils import normalize_scores


class HybridRetriever:
    """Hybrid lexical + semantic retriever.

    The semantic branch can use either DenseRetriever.search() directly or a
    pre-built VectorIndex. Passing a VectorIndex makes the end-to-end path use
    FAISS when it is installed, with sklearn nearest-neighbor fallback otherwise.
    """

    def __init__(self, bm25, dense, alpha: float = 0.55, vector_index=None):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.vector_index = vector_index

    def _dense_search(self, query: str, top_k: int):
        if self.vector_index is not None:
            q_emb = self.dense.encode_queries([query])
            return self.vector_index.search(q_emb, top_k=top_k)[0]
        return self.dense.search(query, top_k)

    def search(self, query: str, top_k: int = 80, candidate_k: int | None = None):
        candidate_k = candidate_k or max(top_k * 3, 100)
        bm = self.bm25.search(query, candidate_k)
        de = self._dense_search(query, candidate_k)
        docs = list(dict.fromkeys([d for d, _ in bm] + [d for d, _ in de]))
        bm_map = dict(bm)
        de_map = dict(de)
        bm_vals = normalize_scores(np.array([bm_map.get(d, 0.0) for d in docs]))
        de_vals = normalize_scores(np.array([de_map.get(d, 0.0) for d in docs]))
        scores = self.alpha * bm_vals + (1 - self.alpha) * de_vals
        doc_keys = np.asarray(docs)
        order = np.lexsort((doc_keys, -scores))[:top_k]
        return [
            (docs[int(i)], float(scores[int(i)]), float(bm_vals[int(i)]), float(de_vals[int(i)]))
            for i in order
        ]
