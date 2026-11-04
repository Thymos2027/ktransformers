#!/usr/bin/env bash
set -Eeuo pipefail
cd /data/deploy/dsv4-20261104
image='quay.io/ascend/sglang@sha256:5ed0acf504181fe092f9badbd8f77611b985ec65df0f93b66ab4b9616231f70b'
test -f /data/llm/DeepSeek-V4-Flash-W8A8/_transfer-verification.json
test -f /data/llm/DeepSeek-V4-Flash/_transfer-verification.json
test -z "$(sudo -n docker ps -aq -f name=^dsv4-kt-20261104$)"
sudo -n docker run -d --name dsv4-kt-20261104 \
  --ipc=host --privileged --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 16g \
  --device /dev/davinci2 --device /dev/davinci5 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --dns 223.5.5.5 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /data/llm:/models:ro \
  -v "$PWD:/workspace" \
  -p 192.168.137.11:18080:18080 \
  --entrypoint /bin/bash "$image" /workspace/pipeline.sh > container.id
cat container.id
