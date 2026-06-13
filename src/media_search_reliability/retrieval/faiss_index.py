from __future__ import annotations

import importlib.util
import numpy as np
from sklearn.neighbors import NearestNeighbors


class VectorIndex:
    def __init__(self, use_gpu_if_available: bool = True):
        self.use_gpu_if_available = use_gpu_if_available
        self.index = None
        self.doc_ids = None
        self.backend = None
        self.nn = None
        self.embeddings = None

    def fit(self, embeddings: np.ndarray, doc_ids: list[int]):
        self.doc_ids = list(doc_ids)
        emb = np.asarray(embeddings, dtype="float32")
        if emb.ndim != 2 or len(emb) != len(self.doc_ids):
            raise ValueError("embeddings must be 2D and aligned with doc_ids")
        if len(emb) == 0:
            raise ValueError("Cannot fit an empty vector index")
        if importlib.util.find_spec("faiss") is not None:
            import faiss
            index = faiss.IndexFlatIP(emb.shape[1])
            if self.use_gpu_if_available and hasattr(faiss, "StandardGpuResources"):
                try:
                    resources = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(resources, 0, index)
                    self.backend = "faiss-gpu"
                except Exception:
                    self.backend = "faiss-cpu"
            else:
                self.backend = "faiss-cpu"
            index.add(emb)
            self.index = index
        else:
            self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self.nn.fit(emb)
            self.embeddings = emb
            self.backend = "sklearn-nearest-neighbors-fallback"
        return self

    def search(self, query_embeddings: np.ndarray, top_k: int = 50):
        q = np.asarray(query_embeddings, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if not self.doc_ids:
            return [[] for _ in range(len(q))]
        k = max(1, min(int(top_k), len(self.doc_ids)))
        if self.index is not None:
            scores, idxs = self.index.search(q, k)
            return [[(self.doc_ids[int(i)], float(s)) for i, s in zip(row_i, row_s) if i >= 0] for row_s, row_i in zip(scores, idxs)]
        distances, idxs = self.nn.kneighbors(q, n_neighbors=k)
        return [[(self.doc_ids[int(i)], float(1 - d)) for i, d in zip(row_i, row_d)] for row_d, row_i in zip(distances, idxs)]
