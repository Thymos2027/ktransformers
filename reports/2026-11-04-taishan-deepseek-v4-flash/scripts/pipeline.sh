#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
mkdir -p logs artifacts sources models
exec > >(tee -a logs/pipeline.log) 2>&1
phase() { python3 /workspace/status.py "$@"; }
fail() { rc="$1"; line="$2"; trap - ERR; phase failed "exit=$rc line=$line; inspect logs/pipeline.log"; exit "$rc"; }
trap 'fail "$?" "$LINENO"' ERR
exec 9>/workspace/pipeline.lock
flock -n 9
phase verifying_weight_records
python3 - <<'PY'
import json
from pathlib import Path
for name,count in [('DeepSeek-V4-Flash',74),('DeepSeek-V4-Flash-W8A8',65)]:
    d=Path('/models')/name
    result=json.loads((d/'_transfer-verification.json').read_text())
    assert result['all_files_sha256_verified'] and result['files']==count,result
    manifest=json.loads((d/'_download-verification.json').read_text())
    assert len(manifest['files'])==count
    for f in manifest['files']:
        p=d/f['path']
        assert p.is_file() and p.stat().st_size==f['size'],p
    print('Verified remote transfer record and current sizes:',name,count)
PY
export http_proxy=http://192.168.137.1:7890 https_proxy=http://192.168.137.1:7890
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" NO_PROXY=127.0.0.1,localhost
export PIP_DEFAULT_TIMEOUT=90 PIP_RETRIES=8 PIP_DISABLE_PIP_VERSION_CHECK=1
phase system_dependencies
export DEBIAN_FRONTEND=noninteractive
if ! pkg-config --exists hwloc || ! command -v patchelf >/dev/null; then
timeout 1800 apt-get -o Acquire::http::Proxy="$http_proxy" -o Acquire::https::Proxy="$https_proxy" -o Acquire::Retries=3 update
timeout 1800 apt-get -o Acquire::http::Proxy="$http_proxy" -o Acquire::https::Proxy="$https_proxy" -o Acquire::Retries=3 install -y build-essential cmake git libhwloc-dev libhwloc15 libnuma-dev patchelf pkg-config curl util-linux
fi
python3 -m pip config --site set global.proxy "$http_proxy"
phase pinned_sources
KT=/workspace/sources/ktransformers
mkdir -p "$KT"
if [ ! -f "$KT/kt-kernel/CMakeLists.txt" ]; then tar -xzf /workspace/ktransformers-c987331.tar.gz -C "$KT"; fi
python3 /workspace/normalize_sources.py
fetch() {
  dir="$1"; url="$2"; commit="$3"
  mkdir -p "$dir"
  if [ ! -d "$dir/.git" ]; then git -C "$dir" init; fi
  git -C "$dir" config http.proxy http://192.168.137.1:7890
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]; then
    success=0
    for attempt in 1 2 3 4; do
      if timeout 900 git -C "$dir" -c http.version=HTTP/1.1 fetch --depth 1 "$url" "$commit"; then success=1; break; fi
      sleep 10
    done
    test "$success" -eq 1
    git -C "$dir" checkout --detach FETCH_HEAD
  fi
  test "$(git -C "$dir" rev-parse HEAD)" = "$commit"
}
fetch /workspace/sources/sglang https://github.com/Pan-Boyi/sglang.git ff46d660d16cbf11ef393c80cc6b3ab7f6259780
fetch "$KT/third_party/llama.cpp" https://github.com/ggml-org/llama.cpp.git a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
fetch "$KT/third_party/pybind11" https://github.com/pybind/pybind11.git bb05e0810b87e74709d9f4c4545f1f57a1b386f5
python3 /workspace/prepare_model_view.py
source /workspace/dsv4.env
source "$DSV4_TOOLS/dsv4_env.sh"
python3 -m pip freeze > /workspace/artifacts/packages-before.txt
phase python_dependencies
timeout 3600 bash "$DSV4_TOOLS/setup.sh" deps
timeout 1200 python3 -m pip install 'setuptools>=61' wheel 'cmake>=3.16' pybind11 ninja safetensors numpy gguf typer rich pyyaml httpx packaging
phase build_kt_kernel
timeout 10800 bash "$DSV4_TOOLS/setup.sh" kt-kernel
phase check_bundled_npu_operators
timeout 10800 bash "$DSV4_TOOLS/setup.sh" sgl-kernel
timeout 10800 bash "$DSV4_TOOLS/setup.sh" cann-ops
phase convert_and_verify_mxfp4_experts
timeout 28800 bash "$DSV4_TOOLS/setup.sh" gguf
phase deployment_preflight
timeout 600 bash "$DSV4_TOOLS/setup.sh" check | tee /workspace/logs/preflight.log
python3 -m pip freeze > /workspace/artifacts/packages-after.txt
phase starting_inference_validation
exec python3 /workspace/run_service.py
