#!/usr/bin/env bash
set -Eeo pipefail
source /workspace/dsv4.env
export SGLANG_REPO=/experiment/sglang
export DSV4_LOG_DIR=/experiment/logs
source "${DSV4_TOOLS}/dsv4_env.sh"
set -u
cd /experiment
case "${1:-}" in
  tp1a|tp1b) tp=1 ;;
  tp2a|tp2b) tp=2 ;;
  *) echo 'Expected tp1a, tp2a, tp2b, or tp1b' >&2; exit 2 ;;
esac
case "$tp" in
  1) export ASCEND_RT_VISIBLE_DEVICES=0; unset KT_NPU_EXPERIMENTAL_TP2 ;;
  2) export ASCEND_RT_VISIBLE_DEVICES=0,1 KT_NPU_EXPERIMENTAL_TP2=1 ;;
  *) exit 2 ;;
esac
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=0
export KT_PREFILL_STREAM=0 KT_DYNAMIC_RESIDENT=0 KT_SIDE_STREAM=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
ulimit -n 65536
python3 -c 'import sglang; print("Benchmark source:", sglang.__file__, flush=True); assert sglang.__file__.startswith("/experiment/sglang/")'
cmd=( python3 -m sglang.launch_server
  --model-path "${DSV4_MODEL_PATH}" --device npu --attention-backend ascend
  --tensor-parallel-size "$tp" --expert-parallel-size 1 --moe-a2a-backend none
  --page-size 128 --quantization compressed-tensors
  --disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code
  --disable-radix-cache --mem-fraction-static 0.81
  --context-length 8192 --max-prefill-tokens 8191 --chunked-prefill-size 512
  --max-total-tokens 16384 --random-seed 20261111
  --watchdog-timeout 600 --kt-method LLAMAFILE --kt-num-gpu-experts 32
  --kt-weight-path "${DSV4_GGUF_TEMPLATE}" --kt-threadpool-count 4 --kt-cpuinfer 96
  --max-running-requests 1 --host 127.0.0.1 --port 18082
)
printf '%s\n' "${cmd[*]}"
exec "${cmd[@]}"
