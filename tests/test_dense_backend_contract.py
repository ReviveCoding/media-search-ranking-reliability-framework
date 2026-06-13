import pytest

from media_search_reliability.retrieval.dense_retriever import DenseBackendError, DenseRetriever


def test_explicit_sentence_transformer_backend_can_be_strict(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "sentence_transformers" else None)
    retriever = DenseRetriever(backend="sentence-transformers", allow_backend_fallback=False)
    with pytest.raises(DenseBackendError, match="explicitly requested"):
        retriever.fit(["one document"], [1])


def test_explicit_backend_fallback_is_labeled(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "sentence_transformers" else None)
    retriever = DenseRetriever(backend="sentence-transformers", allow_backend_fallback=True).fit(
        ["one document", "another document"], [1, 2]
    )
    assert retriever.actual_backend == "tfidf-normalized-fallback"
