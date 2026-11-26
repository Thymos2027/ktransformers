#!/usr/bin/env bash
set -Eeo pipefail
source /workspace/dsv4.env
export SGLANG_REPO=/experiment/sglang DSV4_LOG_DIR=/experiment/logs
source "${DSV4_TOOLS}/dsv4_env.sh"
set -u
cd /experiment
phase="${1:-}"
export ASCEND_RT_VISIBLE_DEVICES=0 SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
export KT_PREFILL_STREAM=1 KT_MXFP4_DEPOOL=1 KT_MXFP4_GGUF_DEDUP=1
export KT_DYNAMIC_RESIDENT=1 KT_SIDE_STREAM=1 KT_PREFILL_STREAM_THRESHOLD=512
export KT_STREAM_WARMUP=1 KT_SHARED_EXPERTS_STREAM=0
export KT_PREFILL_STREAM_CKPT="${DSV4_MODEL_PATH}" KT_MXFP4_CKPT="${DSV4_NATIVE_CKPT}"
export KT_MXFP4_OP_DIR="${KTRANSFORMERS_REPO}/kt-kernel/tools/ascendc_mxfp4"
export KT_MXFP4_SO_DIR=/experiment/mxfp4-kernel KT_GGUF_TEMPLATE="${DSV4_GGUF_TEMPLATE}"
unset KT_NPU_EXPERIMENTAL_TP2 KT_NPU_EXPERIMENTAL_EP2 KT_NPU_EXPERIMENTAL_ATTN_TP1
ulimit -n 65536
python3 -c 'import sglang; print("Tutorial benchmark source:",sglang.__file__,flush=True); assert sglang.__file__.startswith("/experiment/sglang/")'
if [ "$phase" = environment ]; then
  python3 /experiment/environment_probe.py
  exec python3 /experiment/kernel_probe.py
fi
experts=32
fraction=0.81
extra=()
case "$phase" in
  table12) ;;
  table19) extra+=( --swa-full-tokens-ratio 0.35 ) ;;
  table39) experts=28; fraction=0.775 ;;
  hardwarelong)
    experts=28; fraction=0.775
    extra+=( --swa-full-tokens-ratio 0.5 --max-total-tokens 73728 ) ;;
  *) echo 'Unknown phase' >&2; exit 2 ;;
esac
python3 - <<'PY'
import json,os
keys=('KT_PREFILL_STREAM','KT_MXFP4_DEPOOL','KT_MXFP4_GGUF_DEDUP','KT_DYNAMIC_RESIDENT',
      'KT_SIDE_STREAM','KT_PREFILL_STREAM_THRESHOLD','SGLANG_SET_CPU_AFFINITY','ASCEND_RT_VISIBLE_DEVICES')
print('[TUTORIAL CONFIG] '+json.dumps({k:os.environ[k] for k in keys}),flush=True)
PY
cmd=( python3 -m sglang.launch_server
  --model-path "${DSV4_MODEL_PATH}" --device npu --attention-backend ascend
  --tensor-parallel-size 1 --expert-parallel-size 1 --moe-a2a-backend none
  --page-size 128 --quantization compressed-tensors --disable-shared-experts-fusion
  --dtype bfloat16 --trust-remote-code --disable-radix-cache
  --mem-fraction-static "$fraction" --context-length 65536 --max-prefill-tokens 65535
  --chunked-prefill-size 32768 --watchdog-timeout 18000
  --kt-method LLAMAFILE --kt-num-gpu-experts "$experts"
  --kt-weight-path "${DSV4_GGUF_TEMPLATE}" --kt-threadpool-count 1 --kt-cpuinfer 32
  --max-running-requests 1 --enable-metrics --stream-interval 1 --decode-log-interval 100
  --host 127.0.0.1 --port 18082 "${extra[@]}" )
printf '[tutorial launch] %s\n' "${cmd[*]}"
exec "${cmd[@]}"
