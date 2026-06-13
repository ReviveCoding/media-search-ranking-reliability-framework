from __future__ import annotations

import importlib.util
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class DenseBackendError(RuntimeError):
    """Raised when a specifically requested dense backend cannot be initialized."""


class DenseRetriever:
    """SentenceTransformer retrieval with an explicit, labeled TF-IDF fallback."""

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 64,
        n_components: int = 128,
        allow_backend_fallback: bool = True,
        show_progress_bar: bool = False,
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.n_components = n_components
        self.allow_backend_fallback = allow_backend_fallback
        self.show_progress_bar = show_progress_bar
        self.model = None
        self.vectorizer = None
        self.svd = None
        self.embeddings = None
        self.doc_ids = None
        self.actual_backend = None

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _fit_fallback(self, docs: list[str]):
        self.vectorizer = TfidfVectorizer(min_df=1, max_features=2048, ngram_range=(1, 2))
        x = self.vectorizer.fit_transform(docs)
        self.embeddings = normalize(x).astype("float32").toarray()
        self.actual_backend = "tfidf-normalized-fallback"
        return self

    def fit(self, docs: list[str], doc_ids: list[int] | None = None):
        self.doc_ids = doc_ids if doc_ids is not None else list(range(len(docs)))
        requested_st = self.backend in ("auto", "sentence-transformers")
        available = importlib.util.find_spec("sentence_transformers") is not None

        if requested_st and available:
            try:
                from sentence_transformers import SentenceTransformer

                device = self._resolve_device()
                self.model = SentenceTransformer(self.model_name, device=device)
                emb = self.model.encode(
                    docs,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                self.embeddings = emb.astype("float32")
                self.actual_backend = f"sentence-transformers:{device}"
                return self
            except Exception as exc:
                self.model = None
                if self.backend == "sentence-transformers" and not self.allow_backend_fallback:
                    raise DenseBackendError(
                        f"Requested SentenceTransformers backend failed to initialize: {exc}"
                    ) from exc
                warnings.warn(f"SentenceTransformers failed; using labeled TF-IDF fallback: {exc}")
                return self._fit_fallback(docs)

        if self.backend == "sentence-transformers" and not self.allow_backend_fallback:
            raise DenseBackendError(
                "SentenceTransformers was explicitly requested but is not installed. "
                "Install the gpu extra or set retrieval.allow_backend_fallback=true."
            )
        if self.backend not in {"auto", "sentence-transformers", "tfidf"}:
            raise ValueError(f"Unknown dense backend: {self.backend}")
        return self._fit_fallback(docs)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(
                queries,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        if self.vectorizer is None:
            raise DenseBackendError("DenseRetriever must be fitted before encoding queries.")
        x = self.vectorizer.transform(queries)
        return normalize(x).astype("float32").toarray()

    def search(self, query: str, top_k: int = 50):
        q = self.encode_queries([query])[0]
        scores = self.embeddings @ q
        doc_keys = np.asarray(self.doc_ids)
        idx = np.lexsort((doc_keys, -scores))[: min(int(top_k), len(scores))]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]
