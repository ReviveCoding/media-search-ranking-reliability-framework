from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    def __init__(self, method: str = "auto", positive_label_min: int = 2):
        self.method = method
        self.positive_label_min = positive_label_min
        self.model = None
        self.actual_method = "unfitted"
        self._fallback_min = 0.0
        self._fallback_max = 1.0

    def fit(self, scores, labels):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        y = (np.asarray(labels) >= self.positive_label_min).astype(int)
        self._fallback_min = float(np.min(scores)) if len(scores) else 0.0
        self._fallback_max = float(np.max(scores)) if len(scores) else 1.0
        if len(scores) == 0 or len(np.unique(y)) < 2:
            self.model = None
            self.actual_method = "minmax-fallback"
            return self

        method = self.method
        if method == "auto":
            class_counts = np.bincount(y, minlength=2)
            # Isotonic is flexible but can overfit tiny validation sets. Use it
            # only when there are enough observations and score support.
            method = "isotonic" if len(scores) >= 1000 and class_counts.min() >= 30 and len(np.unique(scores)) >= 20 else "platt"

        if method == "platt":
            self.model = LogisticRegression(max_iter=1000, random_state=42).fit(scores.reshape(-1, 1), y)
            self.actual_method = "platt"
        elif method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
            self.actual_method = "isotonic"
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if self.model is None:
            denom = max(1e-12, self._fallback_max - self._fallback_min)
            return np.clip((scores - self._fallback_min) / denom, 0, 1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(scores), dtype=float)
