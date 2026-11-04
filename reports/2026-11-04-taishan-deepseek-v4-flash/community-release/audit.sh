#!/usr/bin/env bash
set -Eeuo pipefail
cd /deploy/dsv4-community-release-20261104
exec > >(tee -a audit.log) 2>&1
export KT_GGUF_PY=/deploy/dsv4-20261104/sources/ktransformers/third_party/llama.cpp/gguf-py
export PYTHONPATH="$KT_GGUF_PY:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
exec nice -n 15 python3 repository/tools/audit_release.py \
  --repository /deploy/dsv4-community-release-20261104/repository \
  --work /deploy/dsv4-community-release-20261104 \
  --gguf-dir /deploy/dsv4-20261104/gguf \
  --source /models/DeepSeek-V4-Flash --jobs 2
