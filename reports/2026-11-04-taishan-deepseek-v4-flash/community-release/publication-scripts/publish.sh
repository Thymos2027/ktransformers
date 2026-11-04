#!/usr/bin/env bash
set -Eeuo pipefail
cd /publish
umask 077
exec > >(tee -a publish.log) 2>&1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/publish/sdk
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=60
export http_proxy=http://192.168.137.1:7890 https_proxy=http://192.168.137.1:7890
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" NO_PROXY=localhost,127.0.0.1
echo 'Installing isolated modelscope-hub 0.6.1; no global credential login.'
python3 -m pip install --target /publish/sdk --upgrade --index-url https://pypi.org/simple --proxy "$https_proxy" 'modelscope-hub==0.6.1'
exec python3 /publish/publish_after_audit.py
