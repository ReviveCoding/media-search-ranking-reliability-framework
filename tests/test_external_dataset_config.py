from pathlib import Path

from media_search_reliability.utils import load_yaml


def test_external_gpu_configs_are_path_override_ready():
    for name in ("pipeline_external_gpu_quick.yaml", "pipeline_external_gpu_full.yaml"):
        cfg = load_yaml(Path("configs") / name)
        assert cfg["data"]["movielens_raw_dir"] is None
        assert cfg["retrieval"]["dense_backend"] == "sentence-transformers"
        assert cfg["ranking"]["device_type"] == "gpu"
        assert cfg["ranking"]["gpu_fallback_to_cpu"] is True
