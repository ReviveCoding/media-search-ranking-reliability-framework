from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _make_config(base: dict, run_root: Path) -> Path:
    cfg = json.loads(json.dumps(base))
    cfg["project"].update({
        "random_seed": 1729,
        "output_dir": str(run_root / "artifacts"),
        "reports_dir": str(run_root / "reports"),
        "data_dir": str(run_root / "data"),
    })
    cfg["data"].update({"demo_movies": 140, "demo_users": 75, "demo_ratings": 1800})
    cfg["queries"].update({"num_queries": 30, "candidates_per_query": 22, "label_all_catalog_demo": True})
    cfg["retrieval"].update({"top_k": 22, "dense_backend": "tfidf", "faiss_use_gpu_if_available": False})
    cfg["ranking"].update({
        "n_estimators": 35,
        "device_type": "cpu",
        "n_jobs": 1,
        "deterministic": True,
        "random_seed": 1729,
    })
    cfg["evaluation"].update({"latency_max_queries": 3, "latency_warmup_queries": 1})
    config_path = run_root / "pipeline.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def _run(config_path: Path, hash_seed: str) -> None:
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": hash_seed,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--config", str(config_path), "--mode", "demo"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=True,
    )


def _stable_summary(summary: dict) -> dict:
    summary = json.loads(json.dumps(summary))
    summary.pop("reports_dir", None)
    summary.pop("artifacts_dir", None)
    metrics = summary.get("metrics", {})
    metrics.pop("p95_latency_ms", None)
    return summary


def _assert_numeric_close(left: object, right: object, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"Key mismatch at {path}: {set(left) ^ set(right)}")
        for key in sorted(left):
            _assert_numeric_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"Length mismatch at {path}")
        for idx, (a, b) in enumerate(zip(left, right)):
            _assert_numeric_close(a, b, f"{path}[{idx}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-12, equal_nan=True):
            raise AssertionError(f"Numeric mismatch at {path}: {left} vs {right}")
        return
    if left != right:
        raise AssertionError(f"Mismatch at {path}: {left!r} vs {right!r}")


def _read_sorted(path: Path, keys: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.sort_values(keys).reset_index(drop=True)


def main() -> None:
    base = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="media_search_repro_") as tmp:
        tmp_root = Path(tmp)
        roots = [tmp_root / "run_a", tmp_root / "run_b"]
        configs = [_make_config(base, root) for root in roots]
        _run(configs[0], "1")
        _run(configs[1], "999")

        summaries = [json.loads((root / "artifacts" / "eval_summary.json").read_text(encoding="utf-8")) for root in roots]
        _assert_numeric_close(_stable_summary(summaries[0]), _stable_summary(summaries[1]))

        query_a = _read_sorted(roots[0] / "data" / "synthetic" / "synthetic_queries.csv", ["query_id"])
        query_b = _read_sorted(roots[1] / "data" / "synthetic" / "synthetic_queries.csv", ["query_id"])
        pd.testing.assert_frame_equal(query_a, query_b, check_exact=True)

        labels_a = _read_sorted(roots[0] / "data" / "processed" / "graded_relevance_labels.csv", ["query_id", "movie_id"])
        labels_b = _read_sorted(roots[1] / "data" / "processed" / "graded_relevance_labels.csv", ["query_id", "movie_id"])
        pd.testing.assert_frame_equal(labels_a, labels_b, check_exact=True)

        pred_a = _read_sorted(roots[0] / "artifacts" / "test_predictions.csv", ["query_id", "movie_id"])
        pred_b = _read_sorted(roots[1] / "artifacts" / "test_predictions.csv", ["query_id", "movie_id"])
        pd.testing.assert_frame_equal(pred_a, pred_b, check_exact=False, rtol=0.0, atol=1e-12)

        ndcg = summaries[0]["metrics"]["ndcg_at_10"]
        lift = summaries[0]["metrics"]["ranker_ndcg_lift_vs_hybrid"]

    report = "\n".join([
        "# Cross-Process Reproducibility Report",
        "",
        "**Decision:** PASS",
        "",
        "The same compact CPU-safe pipeline was executed in two independent Python processes with different `PYTHONHASHSEED` values (1 and 999).",
        "",
        "Verified identical outputs:",
        "",
        "- synthetic queries",
        "- clean and observed graded relevance labels",
        "- split diagnostics",
        "- model-quality metrics excluding wall-clock latency",
        "- row-level test predictions within absolute tolerance `1e-12`",
        "",
        f"Reference NDCG@10: `{ndcg:.6f}`",
        f"Reference ranker lift: `{lift:+.6f}`",
        "",
        "Latency is intentionally excluded because wall-clock measurements are environment dependent.",
    ])
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "18_reproducibility_report.md").write_text(report, encoding="utf-8")
    print("Cross-process reproducibility PASS")


if __name__ == "__main__":
    main()
