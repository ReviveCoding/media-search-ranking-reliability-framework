import numpy as np
from media_search_reliability.retrieval.hybrid_retriever import HybridRetriever


class FakeBM25:
    def search(self, query, top_k=50):
        return [(1, 1.0)]


class FakeDense:
    def encode_queries(self, queries):
        return np.ones((len(queries), 2), dtype="float32")

    def search(self, query, top_k=50):
        return [(2, 1.0)]


class FakeVectorIndex:
    backend = "fake-faiss"

    def search(self, query_embeddings, top_k=50):
        return [[(3, 2.0)]]


def test_hybrid_uses_vector_index_when_available():
    retriever = HybridRetriever(FakeBM25(), FakeDense(), vector_index=FakeVectorIndex())
    docs = [doc_id for doc_id, *_ in retriever.search("robots", top_k=5)]
    assert 3 in docs
    assert 2 not in docs
