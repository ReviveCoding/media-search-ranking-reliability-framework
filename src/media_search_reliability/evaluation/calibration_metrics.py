from __future__ import annotations

import numpy as np


def brier_score(prob, labels, positive_label_min: int = 2) -> float:
    prob = np.asarray(prob, dtype=float)
    y = (np.asarray(labels) >= positive_label_min).astype(float)
    return float(np.mean((prob - y) ** 2))


def expected_calibration_error(prob, labels, positive_label_min: int = 2, n_bins: int = 10) -> float:
    prob = np.asarray(prob, dtype=float)
    y = (np.asarray(labels) >= positive_label_min).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)
