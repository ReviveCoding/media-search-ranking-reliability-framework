from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd

from media_search_reliability.features.retrieval_features import FEATURE_COLUMNS


@dataclass
class RankerBundle:
    model: object
    feature_columns: list[str]
    backend: str


class RankerTrainingError(RuntimeError):
    """Raised when the required learning-to-rank backend cannot be trained."""


def _group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("query_id", sort=False).size().astype(int).tolist()


def _fit_non_ranking_fallback(x_train, y_train, feature_columns: list[str]) -> RankerBundle:
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train, y_train)
    return RankerBundle(model=model, feature_columns=feature_columns, backend="sklearn-gbr-explicit-fallback")


def train_lambdarank(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    config: dict | None = None,
) -> RankerBundle:
    feature_columns = feature_columns or FEATURE_COLUMNS
    config = config or {}
    x_train = train_df[feature_columns].fillna(0.0)
    y_train = train_df["label"].astype(int)
    group_train = _group_sizes(train_df)
    allow_fallback = bool(config.get("allow_non_ranking_fallback", False))

    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        if allow_fallback:
            warnings.warn("LightGBM is unavailable; using explicitly enabled non-ranking fallback.")
            return _fit_non_ranking_fallback(x_train, y_train, feature_columns)
        raise RankerTrainingError(
            "LightGBM is required for this LambdaRank framework. Install project dependencies "
            "or explicitly set ranking.allow_non_ranking_fallback=true for diagnostic-only use."
        ) from exc

    params = {
        "objective": config.get("objective", "lambdarank"),
        "metric": config.get("metric", "ndcg"),
        "n_estimators": int(config.get("n_estimators", 180)),
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "num_leaves": int(config.get("num_leaves", 31)),
        "min_child_samples": int(config.get("min_child_samples", 20)),
        "max_depth": int(config.get("max_depth", -1)),
        "reg_alpha": float(config.get("reg_alpha", 0.0)),
        "reg_lambda": float(config.get("reg_lambda", 0.0)),
        "feature_fraction": float(config.get("feature_fraction", 1.0)),
        "bagging_fraction": float(config.get("bagging_fraction", 1.0)),
        "bagging_freq": int(config.get("bagging_freq", 0)),
        "label_gain": list(config.get("label_gain", [0, 1, 3, 7])),
        "random_state": int(config.get("random_seed", 42)),
        "n_jobs": int(config.get("n_jobs", 1)),
        "deterministic": bool(config.get("deterministic", True)),
        "feature_fraction_seed": int(config.get("random_seed", 42)),
        "bagging_seed": int(config.get("random_seed", 42)),
        "data_random_seed": int(config.get("random_seed", 42)),
        "verbose": -1,
        "force_col_wise": True,
    }
    requested_device = str(config.get("device_type", "auto")).lower()
    if requested_device in {"gpu", "cuda"}:
        params["device_type"] = "gpu"

    fit_kwargs = {"group": group_train}
    if val_df is not None and len(val_df):
        fit_kwargs["eval_set"] = [(val_df[feature_columns].fillna(0.0), val_df["label"].astype(int))]
        fit_kwargs["eval_group"] = [_group_sizes(val_df)]
        fit_kwargs["eval_at"] = tuple(int(x) for x in config.get("eval_at", [5, 10]))
        early_stopping_rounds = int(config.get("early_stopping_rounds", 40))
        if early_stopping_rounds > 0:
            from lightgbm import early_stopping, log_evaluation
            fit_kwargs["callbacks"] = [
                early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                log_evaluation(period=0),
            ]

    model = LGBMRanker(**params)
    try:
        model.fit(x_train, y_train, **fit_kwargs)
        backend = "lightgbm-lambdarank-gpu" if params.get("device_type") == "gpu" else "lightgbm-lambdarank-cpu"
        return RankerBundle(model=model, feature_columns=feature_columns, backend=backend)
    except Exception as gpu_exc:
        if params.get("device_type") == "gpu" and bool(config.get("gpu_fallback_to_cpu", True)):
            warnings.warn(f"LightGBM GPU training failed, retrying CPU LambdaRank: {gpu_exc}")
            cpu_params = dict(params)
            cpu_params.pop("device_type", None)
            cpu_model = LGBMRanker(**cpu_params)
            try:
                cpu_model.fit(x_train, y_train, **fit_kwargs)
                return RankerBundle(
                    model=cpu_model,
                    feature_columns=feature_columns,
                    backend="lightgbm-lambdarank-cpu-fallback",
                )
            except Exception as cpu_exc:
                if allow_fallback:
                    warnings.warn(f"CPU LambdaRank failed; using explicitly enabled non-ranking fallback: {cpu_exc}")
                    return _fit_non_ranking_fallback(x_train, y_train, feature_columns)
                raise RankerTrainingError("CPU LambdaRank training failed after GPU fallback.") from cpu_exc
        if allow_fallback:
            warnings.warn(f"LambdaRank failed; using explicitly enabled non-ranking fallback: {gpu_exc}")
            return _fit_non_ranking_fallback(x_train, y_train, feature_columns)
        raise RankerTrainingError("LambdaRank training failed; refusing silent non-ranking fallback.") from gpu_exc


def predict_ranker(bundle: RankerBundle, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(bundle.model.predict(df[bundle.feature_columns].fillna(0.0)), dtype=float)
