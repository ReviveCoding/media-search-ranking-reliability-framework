from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import os
from pathlib import Path


def _version(module_name: str):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def collect_environment() -> dict:
    payload = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "sklearn": _version("sklearn"),
            "lightgbm": _version("lightgbm"),
            "faiss": _version("faiss"),
            "sentence_transformers": _version("sentence_transformers"),
            "torch": _version("torch"),
            "fastapi": _version("fastapi"),
        },
        "gpu": {
            "torch_cuda_available": False,
            "torch_cuda_device_count": 0,
            "torch_cuda_device_name_0": None,
            "faiss_gpu_symbols_available": False,
            "lightgbm_gpu_requested_by_default_config": False,
            "gpu_profile_available": Path("configs/pipeline_gpu.yaml").exists(),
            "lightgbm_gpu_requested_by_gpu_profile": False,
        },
        "recommendations": [],
    }
    try:
        import torch
        payload["gpu"]["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu"]["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        payload["gpu"]["torch_cuda_device_name_0"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        payload["gpu"]["torch_error"] = str(e)

    if importlib.util.find_spec("faiss") is not None:
        try:
            import faiss
            payload["gpu"]["faiss_gpu_symbols_available"] = bool(hasattr(faiss, "StandardGpuResources"))
        except Exception as e:
            payload["gpu"]["faiss_error"] = str(e)

    config_path = Path("configs/pipeline.yaml")
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        payload["gpu"]["lightgbm_gpu_requested_by_default_config"] = "device_type: gpu" in text
    gpu_config_path = Path("configs/pipeline_gpu.yaml")
    if gpu_config_path.exists():
        text = gpu_config_path.read_text(encoding="utf-8")
        payload["gpu"]["lightgbm_gpu_requested_by_gpu_profile"] = "device_type: gpu" in text


    external_raw = os.environ.get("MOVIELENS_RAW_DIR")
    payload["dataset"] = {
        "movielens_raw_dir_env": external_raw,
        "movielens_path_valid": None,
        "movielens_resolved_path": None,
        "movielens_path_error": None,
    }
    if external_raw:
        try:
            from media_search_reliability.data_ingestion.load_movielens import resolve_movielens_directory

            resolved = resolve_movielens_directory(external_raw)
            payload["dataset"]["movielens_path_valid"] = True
            payload["dataset"]["movielens_resolved_path"] = str(resolved)
        except Exception as exc:
            payload["dataset"]["movielens_path_valid"] = False
            payload["dataset"]["movielens_path_error"] = str(exc)
            payload["recommendations"].append(f"Fix MOVIELENS_RAW_DIR before dataset mode: {exc}")

    if payload["packages"]["sentence_transformers"] is None:
        payload["recommendations"].append("Install sentence-transformers to enable transformer dense embeddings and CUDA inference when torch CUDA is available.")
    if payload["packages"]["faiss"] is None:
        if platform.system() == "Windows":
            payload["recommendations"].append("FAISS wheels are platform-dependent on Windows. The framework remains runnable with the labeled sklearn vector-index fallback; use a compatible Conda/WSL FAISS build when desired.")
        else:
            payload["recommendations"].append("Install the optional faiss extra or a compatible FAISS GPU build to replace sklearn nearest-neighbor fallback.")
    if payload["packages"]["lightgbm"] is None:
        payload["recommendations"].append("Install lightgbm; the default ranker contract refuses a silent non-ranking fallback.")
    if payload["gpu"]["gpu_profile_available"]:
        payload["recommendations"].append("Use configs/pipeline_gpu.yaml for the explicit GPU-oriented training/inference path.")
    if not payload["gpu"]["torch_cuda_available"]:
        payload["recommendations"].append("CUDA was not detected in this environment. The project will still run on CPU fallback.")
    return payload


def write_report(payload: dict, path: Path) -> None:
    lines = ["# Environment and GPU Readiness Report", ""]
    lines.append("## Runtime")
    lines.append(f"- Python: {payload['python_version']}")
    lines.append(f"- Platform: {payload['platform']}")
    lines.append("")
    lines.append("## Package availability")
    for k, v in payload["packages"].items():
        lines.append(f"- {k}: {v if v is not None else 'not installed'}")
    lines.append("")
    lines.append("## GPU readiness")
    for k, v in payload["gpu"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Dataset readiness")
    for k, v in payload.get("dataset", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Recommendations")
    if payload["recommendations"]:
        for rec in payload["recommendations"]:
            lines.append(f"- {rec}")
    else:
        lines.append("- Environment looks ready for the advanced optional path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="artifacts/environment_report.json")
    parser.add_argument("--md-out", default="reports/12_environment_and_runnability_report.md")
    args = parser.parse_args()
    payload = collect_environment()
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, Path(args.md_out))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
