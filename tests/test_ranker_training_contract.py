import pandas as pd
import pytest

from media_search_reliability.features.retrieval_features import FEATURE_COLUMNS
from media_search_reliability.ranking.train_lambdarank import RankerTrainingError, train_lambdarank


def _tiny_frame():
    rows = []
    for query_id in (1, 2):
        for movie_id, label in enumerate((3, 2, 0), start=1):
            row = {name: float(movie_id) / 3 for name in FEATURE_COLUMNS}
            row.update({"query_id": query_id, "movie_id": movie_id, "label": label})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("query_id")


def test_cpu_lambdarank_failure_is_not_silently_replaced(monkeypatch):
    import lightgbm

    class BrokenRanker:
        def __init__(self, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            raise ValueError("synthetic training failure")

    monkeypatch.setattr(lightgbm, "LGBMRanker", BrokenRanker)
    frame = _tiny_frame()
    with pytest.raises(RankerTrainingError, match="refusing silent"):
        train_lambdarank(frame, config={"device_type": "cpu", "allow_non_ranking_fallback": False})


def test_non_ranking_fallback_requires_explicit_opt_in(monkeypatch):
    import lightgbm

    class BrokenRanker:
        def __init__(self, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            raise ValueError("synthetic training failure")

    monkeypatch.setattr(lightgbm, "LGBMRanker", BrokenRanker)
    frame = _tiny_frame()
    bundle = train_lambdarank(frame, config={"device_type": "cpu", "allow_non_ranking_fallback": True})
    assert bundle.backend == "sklearn-gbr-explicit-fallback"
