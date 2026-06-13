from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path


def _run_with_config(config: str | None, mode: str) -> dict:
    from media_search_reliability.pipeline import run_pipeline

    if config is not None:
        return run_pipeline(config, mode=mode)
    local_default = Path("configs/pipeline.yaml")
    if local_default.exists():
        return run_pipeline(local_default, mode=mode)
    resource = resources.files("media_search_reliability.configs").joinpath("pipeline.yaml")
    with resources.as_file(resource) as packaged_config:
        return run_pipeline(packaged_config, mode=mode)


def pipeline_main() -> None:
    parser = argparse.ArgumentParser(description="Run the media search reliability pipeline.")
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline YAML. Defaults to ./configs/pipeline.yaml, then the packaged CPU-safe template.",
    )
    parser.add_argument("--mode", choices=["demo", "movielens"], default="demo")
    args = parser.parse_args()
    summary = _run_with_config(args.config, args.mode)
    print("Pipeline complete.")
    print(f"launch_decision: {summary['launch_decision']}")
    print(f"ranker_backend: {summary['ranker_backend']}")
    print(f"metrics: {summary['metrics']}")


def download_main() -> None:
    from media_search_reliability.data_ingestion.download_utils import (
        URLS,
        download,
        flatten_movielens_directory,
        safe_extract,
        sha256_file,
    )

    parser = argparse.ArgumentParser(description="Download and safely extract a MovieLens dataset.")
    parser.add_argument("--variant", choices=URLS.keys(), default="1m")
    parser.add_argument("--out", default="data/raw/movielens")
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"ml-{args.variant}.zip"
    download(URLS[args.variant], zip_path)
    actual = sha256_file(zip_path)
    print(f"SHA256: {actual}")
    if args.sha256 and actual.lower() != args.sha256.lower():
        zip_path.unlink(missing_ok=True)
        raise SystemExit("Downloaded file checksum did not match --sha256.")
    safe_extract(zip_path, out)
    flatten_movielens_directory(out)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    print(f"Files available in {out}")


def check_path_main() -> None:
    import json
    from media_search_reliability.data_ingestion.load_movielens import resolve_movielens_directory

    parser = argparse.ArgumentParser(description="Validate an external MovieLens directory.")
    parser.add_argument("path")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    requested = Path(args.path).expanduser()
    resolved = resolve_movielens_directory(requested)
    layout = "dat" if (resolved / "movies.dat").exists() else "csv"
    names = ["movies.dat", "ratings.dat", "tags.dat"] if layout == "dat" else ["movies.csv", "ratings.csv", "tags.csv"]
    files = {}
    for name in names:
        candidate = resolved / name
        files[name] = {
            "exists": candidate.exists(),
            "size_mb": round(candidate.stat().st_size / (1024 * 1024), 3) if candidate.exists() else None,
        }
    payload = {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "layout": layout,
        "files": files,
        "runnable": files[names[0]]["exists"] and files[names[1]]["exists"],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if not payload["runnable"]:
        raise SystemExit(2)
