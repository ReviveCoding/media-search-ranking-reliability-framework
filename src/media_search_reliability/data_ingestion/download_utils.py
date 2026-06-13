from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import zipfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

URLS = {
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(
    zip_path: Path,
    destination: Path,
    *,
    max_members: int = 100_000,
    max_uncompressed_bytes: int = 5 * 1024**3,
) -> None:
    """Extract a zip after path, corruption, member-count, and size checks."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt zip member: {corrupt}")
        members = archive.infolist()
        if len(members) > max_members:
            raise ValueError(f"Archive has too many members: {len(members)} > {max_members}")
        total_size = sum(max(0, int(member.file_size)) for member in members)
        if total_size > max_uncompressed_bytes:
            raise ValueError(
                f"Archive expands beyond allowed size: {total_size} > {max_uncompressed_bytes} bytes"
            )
        for member in members:
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {member.filename}") from exc
        archive.extractall(destination)


def download(url: str, path: Path, timeout: int = 60, user_agent: str = "media-search-reliability/0.5") -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS dataset downloads are allowed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def flatten_movielens_directory(out: Path) -> None:
    subdirs = sorted(path for path in out.iterdir() if path.is_dir() and path.name.startswith("ml-"))
    for subdir in subdirs:
        for source in sorted(subdir.iterdir(), key=lambda p: p.name):
            target = out / source.name
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            shutil.move(str(source), str(target))
        shutil.rmtree(subdir, ignore_errors=True)
