from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path("scripts/run_frozen_movielens_ablation.py")
spec = importlib.util.spec_from_file_location("frozen_ablation_script", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gpu_scale_replay_delta_is_tolerated() -> None:
    original = pd.Series({
        "ndcg_at_10": 0.3220000,
        "mrr_at_10": 0.5460000,
        "recall_efficiency_at_10": 0.2660000,
    })
    replay = pd.Series({
        "ndcg_at_10": 0.3220004,
        "mrr_at_10": 0.5459995,
        "recall_efficiency_at_10": 0.2660003,
    })
    result = module._replay_diagnostics(original, replay, atol=1e-3, rtol=1e-3)
    assert result["passed"] is True


def test_material_replay_delta_is_rejected() -> None:
    original = pd.Series({
        "ndcg_at_10": 0.322,
        "mrr_at_10": 0.546,
        "recall_efficiency_at_10": 0.266,
    })
    replay = pd.Series({
        "ndcg_at_10": 0.310,
        "mrr_at_10": 0.530,
        "recall_efficiency_at_10": 0.240,
    })
    result = module._replay_diagnostics(original, replay, atol=1e-3, rtol=1e-3)
    assert result["passed"] is False
