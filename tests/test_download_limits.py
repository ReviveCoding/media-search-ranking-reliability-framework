from pathlib import Path
import zipfile

import pytest

from media_search_reliability.data_ingestion.download_utils import safe_extract


def test_safe_extract_rejects_member_count_limit(tmp_path: Path):
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.txt", "a")
        handle.writestr("b.txt", "b")
    with pytest.raises(ValueError, match="too many members"):
        safe_extract(archive, tmp_path / "out", max_members=1)


def test_safe_extract_rejects_uncompressed_size_limit(tmp_path: Path):
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("large.txt", "x" * 32)
    with pytest.raises(ValueError, match="allowed size"):
        safe_extract(archive, tmp_path / "out", max_uncompressed_bytes=8)
