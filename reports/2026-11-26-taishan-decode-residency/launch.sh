#!/usr/bin/env bash
set -Eeo pipefail
source /workspace/dsv4.env
export SGLANG_REPO=/experiment/sglang DSV4_LOG_DIR=/experiment/logs
source "${DSV4_TOOLS}/dsv4_env.sh"
set -u
cd /experiment
phase="${1:-}"
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_SET_CPU_AFFINITY=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
export KT_PREFILL_STREAM=1 KT_DYNAMIC_RESIDENT=1 KT_SIDE_STREAM=1
export KT_SHARED_EXPERTS_STREAM=0 KT_MXFP4_DEPOOL=1 KT_MXFP4_GGUF_DEDUP=1
export KT_PREFILL_STREAM_THRESHOLD=512 KT_STREAM_WARMUP=0 KT_HOT_TAIL_TOKENS=0
export KT_PREFILL_STREAM_CKPT="${DSV4_MODEL_PATH}" KT_MXFP4_CKPT="${DSV4_NATIVE_CKPT}"
export KT_MXFP4_OP_DIR="${KTRANSFORMERS_REPO}/kt-kernel/tools/ascendc_mxfp4"
export KT_MXFP4_SO_DIR=/experiment/mxfp4-kernel KT_GGUF_TEMPLATE="${DSV4_GGUF_TEMPLATE}"
export KT_MXFP4_PREFETCH=1 KT_MXFP4_COPY_THREADS=32 KT_MXFP4_BLK_KERNEL=1
export KT_RESIDENCY_INTERVAL=32 KT_RESIDENCY_BUDGET=8 KT_RESIDENCY_CACHE_MIB=1024
export KT_DECODE_RESIDENCY=0 PYTHONPATH=/experiment:${PYTHONPATH}
unset KT_NPU_EXPERIMENTAL_TP2 KT_NPU_EXPERIMENTAL_EP2 KT_NPU_EXPERIMENTAL_ATTN_TP1
unset KT_MOE_PHASE_TIMING KT_MOE_PHASE_TIMING_INTERVAL KT_DECODE_TIMING
case "$phase" in
  environment|a0|a1) ;;
  b0|b1) export KT_DECODE_RESIDENCY=1 ;;
  *) echo 'Unknown residency phase' >&2; exit 2 ;;
esac
export RESIDENCY_PHASE="$phase"
ulimit -n 65536
python3 - <<'PY'
import json, os, sglang
assert sglang.__file__.startswith('/experiment/sglang/'), sglang.__file__
keys=('KT_PREFILL_STREAM','KT_DYNAMIC_RESIDENT','KT_SIDE_STREAM','KT_MXFP4_DEPOOL',
      'KT_MXFP4_GGUF_DEDUP','KT_STREAM_WARMUP','KT_DECODE_RESIDENCY','KT_RESIDENCY_INTERVAL',
      'KT_RESIDENCY_BUDGET','KT_RESIDENCY_CACHE_MIB','TASK_QUEUE_ENABLE')
print('[RESIDENCY CONFIG] '+json.dumps({k:os.environ[k] for k in keys},sort_keys=True),flush=True)
PY
if [ "$phase" = environment ]; then
  python3 /experiment/environment_probe.py
  python3 /experiment/kernel_probe.py
  exec python3 /experiment/hardware_gate.py
fi
exec python3 -m sglang.launch_server \
  --model-path "${DSV4_MODEL_PATH}" --device npu --attention-backend ascend \
  --tensor-parallel-size 1 --expert-parallel-size 1 --data-parallel-size 1 \
  --moe-a2a-backend none --page-size 128 --quantization compressed-tensors \
  --disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code \
  --disable-radix-cache --mem-fraction-static 0.81 \
  --context-length 8192 --max-prefill-tokens 8191 --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.5 --max-total-tokens 16384 \
  --random-seed 20261111 --watchdog-timeout 600 \
  --kt-method LLAMAFILE --kt-num-gpu-experts 32 \
  --kt-weight-path "${DSV4_GGUF_TEMPLATE}" --kt-threadpool-count 4 --kt-cpuinfer 96 \
  --max-running-requests 1 --enable-metrics --stream-interval 1 --decode-log-interval 20 \
  --host 127.0.0.1 --port 18082
