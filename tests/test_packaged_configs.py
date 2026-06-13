from importlib import resources
from pathlib import Path


def test_packaged_configs_match_repository_templates():
    packaged = resources.files("media_search_reliability.configs")
    for name in ("pipeline.yaml", "pipeline_gpu.yaml", "monte_carlo.yaml"):
        repo_text = Path("configs", name).read_text(encoding="utf-8")
        package_text = packaged.joinpath(name).read_text(encoding="utf-8")
        assert package_text == repo_text


def test_pipeline_cli_can_resolve_packaged_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resource = resources.files("media_search_reliability.configs").joinpath("pipeline.yaml")
    assert resource.is_file()
