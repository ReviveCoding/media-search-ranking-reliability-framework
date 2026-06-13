from __future__ import annotations

import argparse
from pathlib import Path

from media_search_reliability.data_ingestion.download_utils import (
    URLS,
    download,
    flatten_movielens_directory,
    safe_extract,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and safely extract a MovieLens dataset.")
    parser.add_argument("--variant", choices=URLS.keys(), default="1m")
    parser.add_argument("--out", default="data/raw/movielens")
    parser.add_argument("--sha256", default=None, help="Optional expected SHA256 checksum.")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"ml-{args.variant}.zip"
    print(f"Downloading {URLS[args.variant]} to {zip_path}")
    download(URLS[args.variant], zip_path)
    actual_sha256 = sha256_file(zip_path)
    print(f"SHA256: {actual_sha256}")
    if args.sha256 and actual_sha256.lower() != args.sha256.lower():
        zip_path.unlink(missing_ok=True)
        raise SystemExit("Downloaded file checksum did not match --sha256.")

    safe_extract(zip_path, out)
    flatten_movielens_directory(out)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    print(f"Done. Files available in {out}")


if __name__ == "__main__":
    main()
