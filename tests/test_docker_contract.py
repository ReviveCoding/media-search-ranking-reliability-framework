from pathlib import Path


def test_dockerfile_includes_packaging_metadata_before_install():
    text = Path("Dockerfile").read_text(encoding="utf-8")
    copy_pos = text.index("COPY pyproject.toml README.md LICENSE ./")
    install_pos = text.index("RUN python -m pip install --no-cache-dir .")
    assert copy_pos < install_pos
    assert 'CMD ["media-search-run"' in text
