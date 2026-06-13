from __future__ import annotations

from collections import Counter
import math
import numpy as np
from media_search_reliability.utils import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter = Counter()
        self.avgdl = 0.0
        self.doc_ids: list[int] = []

    def fit(self, docs: list[str], doc_ids: list[int] | None = None):
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_ids = doc_ids if doc_ids is not None else list(range(len(docs)))
        self.avgdl = float(np.mean([len(t) for t in self.doc_tokens])) if self.doc_tokens else 0.0
        self.doc_freq = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.doc_freq[t] += 1
        return self

    def score(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        n_docs = len(self.doc_tokens)
        scores = np.zeros(n_docs, dtype=float)
        if n_docs == 0:
            return scores
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            dl = len(toks) or 1
            total = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                total += idf * (tf[term] * (self.k1 + 1)) / denom
            scores[i] = total
        return scores

    def search(self, query: str, top_k: int = 50):
        scores = self.score(query)
        if len(scores) == 0:
            return []
        doc_keys = np.asarray(self.doc_ids)
        idx = np.lexsort((doc_keys, -scores))[:top_k]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]
