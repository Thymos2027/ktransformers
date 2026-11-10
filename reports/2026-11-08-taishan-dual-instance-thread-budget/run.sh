#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/dsv4.env
export DSV4_NPU_DEVICE_ID=1
export DSV4_PORT=18081
export DSV4_HOST=0.0.0.0
export DSV4_CPUINFER=32
export DSV4_LOG_DIR=/workspace/dual-instance-thread-budget-20261110/logs
export DSV4_EXTRA_FLAGS=
cd /workspace/dual-instance-thread-budget-20261110
exec python3 supervisor.py
