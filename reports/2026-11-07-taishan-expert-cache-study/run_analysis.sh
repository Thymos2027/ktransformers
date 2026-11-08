#!/usr/bin/env bash
set -euo pipefail
cd /workspace/expert-cache-study-20261107
echo running > analysis-status.txt
trap 'echo failed > analysis-status.txt' ERR
trace=$(find traces -name 'expert_distribution_recorder_*.pt' -type f | sort | tail -1)
python3 replay.py "$trace" --output replay-detail.json > logs/replay-detail.log 2>&1
python3 measure_transfer.py > logs/transfer.log 2>&1
echo complete > analysis-status.txt
