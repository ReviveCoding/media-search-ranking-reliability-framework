from __future__ import annotations

import numpy as np

from media_search_reliability.ranking.calibrate_scores import ScoreCalibrator


def test_small_calibration_set_uses_platt() -> None:
    scores = np.linspace(-2.0, 2.0, 400)
    labels = np.array(([0] * 200) + ([3] * 200))
    calibrator = ScoreCalibrator(method="auto").fit(scores, labels)
    assert calibrator.actual_method == "platt"


def test_large_calibration_set_can_use_isotonic() -> None:
    scores = np.linspace(-3.0, 3.0, 1_200)
    labels = np.array(([0] * 600) + ([3] * 600))
    calibrator = ScoreCalibrator(method="auto").fit(scores, labels)
    assert calibrator.actual_method == "isotonic"
