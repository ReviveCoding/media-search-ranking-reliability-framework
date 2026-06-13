from pathlib import Path
import zipfile

import pytest

from media_search_reliability.data_ingestion.download_utils import safe_extract


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe zip member"):
        safe_extract(archive, tmp_path / "out")
