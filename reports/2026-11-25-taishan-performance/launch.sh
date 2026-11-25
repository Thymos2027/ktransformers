#!/usr/bin/env bash
set -Eeo pipefail
source /workspace/dsv4.env
export SGLANG_REPO=/experiment/sglang DSV4_LOG_DIR=/experiment/logs
source "${DSV4_TOOLS}/dsv4_env.sh"
set -u
cd /experiment
dpa=1; mode=graph; wide=0
case "${1:-}" in
  environment) mode=environment ;;
  hccl) mode=hccl ;;
  dpaeager) mode=eager ;;
  dpagraph) ;;
  epgraph) dpa=0 ;;
  epprefill) dpa=0; wide=1 ;;
  dpaprefill) wide=1 ;;
  *) echo 'Unknown performance phase' >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES=0,1 KT_NPU_EXPERIMENTAL_EP2=1
if [ "$dpa" = 1 ]; then export KT_NPU_EXPERIMENTAL_ATTN_TP1=1
else unset KT_NPU_EXPERIMENTAL_ATTN_TP1; fi
export SGLANG_DP_USE_GATHERV=0 SGLANG_DP_USE_REDUCE_SCATTER=0 SGLANG_DP_SHARED_EXPERT_LOCAL=0
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 SGLANG_SET_CPU_AFFINITY=0
export KT_PREFILL_STREAM=0 KT_DYNAMIC_RESIDENT=0 KT_SIDE_STREAM=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
export PYTHONPATH=/experiment:${PYTHONPATH}
ulimit -n 65536
python3 -c 'import sglang; print("Performance source:", sglang.__file__, flush=True); assert sglang.__file__.startswith("/experiment/sglang/")'
if [ "$mode" = hccl ]; then exec python3 /experiment/hccl_probe.py; fi
if [ "$mode" = environment ]; then
  python3 /experiment/environment_probe.py
  python3 /experiment/request_budget_probe.py
  exec python3 /experiment/device_probe.py
fi
python3 /experiment/environment_probe.py
chunk=512; ratio=0.1
if [ "$wide" = 1 ]; then chunk=8192; ratio=0.5; fi
if [ "$dpa" = 1 ]; then chunk=$((chunk * 2)); fi
cmd=( python3 -m sglang.launch_server
  --model-path "${DSV4_MODEL_PATH}" --device npu --attention-backend ascend
  --tensor-parallel-size 2 --expert-parallel-size 2 --moe-a2a-backend none
  --page-size 128 --quantization compressed-tensors
  --disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code
  --disable-radix-cache --mem-fraction-static 0.81
  --context-length 8192 --max-prefill-tokens 8191 --chunked-prefill-size "$chunk"
  --swa-full-tokens-ratio "$ratio" --max-total-tokens 16384
  --random-seed 20261111 --watchdog-timeout 600
  --kt-method LLAMAFILE --kt-num-gpu-experts 64
  --kt-weight-path "${DSV4_GGUF_TEMPLATE}" --kt-threadpool-count 4 --kt-cpuinfer 96
  --enable-metrics --stream-interval 1 --decode-log-interval 20
  --host 127.0.0.1 --port 18082
)
if [ "$dpa" = 1 ]; then
  cmd+=( --data-parallel-size 2 --enable-dp-attention --max-running-requests 2 )
else
  cmd+=( --data-parallel-size 1 --max-running-requests 1 )
fi
if [ "$mode" = eager ]; then cmd+=( --disable-cuda-graph ); fi
printf '%s\n' "${cmd[*]}"
exec "${cmd[@]}"
