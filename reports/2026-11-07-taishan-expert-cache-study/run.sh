#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/dsv4.env
cd /workspace/expert-cache-study-20261107
exec python3 collect.py >> collector.log 2>&1
