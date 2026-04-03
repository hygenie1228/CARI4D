#!/usr/bin/env bash
set -euo pipefail

# ===== config =====
ENV_PY="/home/namhj/miniconda3/envs/cari4d/bin/python"
ENV_PIP="/home/namhj/miniconda3/envs/cari4d/bin/pip"
# RTX A6000 = sm_86
export TORCH_CUDA_ARCH_LIST="8.6"
# 필요 시 명시 (시스템에 맞게 수정)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "[1/6] Python/PyTorch 확인"
"$ENV_PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "[2/6] 기존 nvdiffrast 제거"
"$ENV_PIP" uninstall -y nvdiffrast || true

echo "[3/6] 빌드 도구 업데이트"
"$ENV_PIP" install -U pip setuptools wheel ninja

echo "[4/6] nvdiffrast 소스 설치 (빌드 격리 비활성화)"
"$ENV_PIP" install --no-cache-dir --no-build-isolation --no-binary nvdiffrast \
  "git+https://github.com/NVlabs/nvdiffrast.git"

echo "[5/6] 설치 정보 확인"
"$ENV_PIP" show nvdiffrast

echo "[6/6] 런타임 검증 (RasterizeCudaContext)"
"$ENV_PY" - <<'PY'
import nvdiffrast, nvdiffrast.torch as dr
print("nvdiffrast:", getattr(nvdiffrast, "__version__", "unknown"))
ctx = dr.RasterizeCudaContext()
print("RasterizeCudaContext OK:", type(ctx))
PY

echo "완료: nvdiffrast 재빌드/검증 성공"