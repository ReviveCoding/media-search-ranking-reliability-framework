# Environment and GPU Readiness Report

## Runtime
- Python: 3.11.9
- Platform: Windows-10-10.0.26200-SP0

## Package availability
- numpy: 2.4.4
- pandas: 3.0.3
- sklearn: 1.9.0
- lightgbm: 4.6.0
- faiss: not installed
- sentence_transformers: 5.5.1
- torch: 2.11.0+cu128
- fastapi: 0.136.3

## GPU readiness
- torch_cuda_available: True
- torch_cuda_device_count: 1
- torch_cuda_device_name_0: NVIDIA GeForce RTX 4090 Laptop GPU
- faiss_gpu_symbols_available: False
- lightgbm_gpu_requested_by_default_config: False
- gpu_profile_available: True
- lightgbm_gpu_requested_by_gpu_profile: True

## Dataset readiness
- movielens_raw_dir_env: C:\Users\bjw-0\Downloads\Project_Data\ml-10m
- movielens_path_valid: True
- movielens_resolved_path: C:\Users\bjw-0\Downloads\Project_Data\ml-10m
- movielens_path_error: None

## Recommendations
- FAISS wheels are platform-dependent on Windows. The framework remains runnable with the labeled sklearn vector-index fallback; use a compatible Conda/WSL FAISS build when desired.
- Use configs/pipeline_gpu.yaml for the explicit GPU-oriented training/inference path.
