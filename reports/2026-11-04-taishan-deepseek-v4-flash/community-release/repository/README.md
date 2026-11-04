---
license: mit
tags:
  - deepseek-v4
  - gguf
  - mxfp4
  - ktransformers
  - cpu-offload
---

# DeepSeek-V4-Flash — community MXFP4 CPU experts for KTransformers

本社区转换包包含 **43 个逐层 MXFP4 CPU 专家 GGUF**，从官方 `deepseek-ai/DeepSeek-V4-Flash` 原始权重提取和重排。它是社区转换版，并非 DeepSeek、KTransformers 或硬件厂商的官方发布。

**本包只有 CPU 路由专家，不能独立运行。** 已验证的 Ascend 路径还需要 `sgl-npu/DeepSeek-V4-Flash-W8A8` 提供 NPU 注意力、稠密层及常驻专家，以及匹配的 KT/SGLang 与算子环境。本包不分发 W8A8。

## 文件与来源

- `dsv4_layer0_mxfp4.gguf` 至 `dsv4_layer42_mxfp4.gguf`，共 43 个。
- 每文件 3,422,552,640 字节；合计 147,169,763,520 字节，约 147.17 GB / 137.06 GiB。
- GGML MXFP4 类型 ID 为 39；专家专用布局，不是通用 GGUF 推理器可以直接运行的整模型。
- 保留原始 MXFP4 专家编码与缩放，重排打包布局，不重新量化为另一种 4-bit 格式。
- `SHA256SUMS` 覆盖 43 个权重，服务器完整审计完成后确认；本地草稿不含权重。
- 来源以 [provenance.json](provenance.json)、[source-files.json](source-files.json) 中的 immutable revision、大小、SHA-256 为准。

| 项目 | 固定版本 |
| --- | --- |
| 原始模型 | [deepseek-ai/DeepSeek-V4-Flash / ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) |
| 原始许可证 | DeepSeek MIT，版权声明保留在 `LICENSE` |
| KT 转换工具 | `c9873312643b615a717832946fba0346e69ab035`，`kt-kernel/tools/mxfp4_gguf` |
| llama.cpp | `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f` 加 KT 自带补丁 |
| SGLang | `Pan-Boyi/sglang`，`ff46d660d16cbf11ef393c80cc6b3ab7f6259780` |
| 配套 W8A8 | [sgl-npu/DeepSeek-V4-Flash-W8A8](https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8)，`compressed-tensors` |

`tools/` 中的 KT 脚本另按 Apache-2.0 分发，许可证为 `LICENSE.tools`。模型数据的 MIT 与工具许可证分别保留。

## 转换复现

需要已校验的原始 checkpoint、对应 KT 提交、应用 KT 补丁的 llama.cpp `gguf-py`，以及 Python、torch、numpy、safetensors。KT 构建步骤为固定 llama.cpp 提交应用补丁，参见该提交的 Ascend 教程及 `setup.sh kt-kernel`。不要使用不认识 MXFP4 类型 39 的 reader/writer。

```bash
export KTRANSFORMERS_REPO=/path/to/ktransformers
export SOURCE_CKPT=/path/to/DeepSeek-V4-Flash
export EXPERT_GGUF_DIR=/path/to/expert-gguf
export KT_GGUF_PY="$KTRANSFORMERS_REPO/third_party/llama.cpp/gguf-py"

python3 "$KTRANSFORMERS_REPO/kt-kernel/tools/mxfp4_gguf/convert_mxfp4_gguf.py" batch \
  --input "$SOURCE_CKPT" --output-dir "$EXPERT_GGUF_DIR" \
  --layer-start 0 --layer-end 42 --jobs 8 --skip-existing

cd "$EXPERT_GGUF_DIR"
sha256sum -c SHA256SUMS

# set --deep 43 仍只抽查少量专家；完整检查须显式使用 256。
for layer in $(seq 0 42); do
  python3 "$KTRANSFORMERS_REPO/kt-kernel/tools/mxfp4_gguf/verify_mxfp4_gguf.py" layer \
    --gguf "$EXPERT_GGUF_DIR/dsv4_layer${layer}_mxfp4.gguf" \
    --model-dir "$SOURCE_CKPT" --layer-idx "$layer" --n-experts-check 256 || exit 1
done
```

不同工具版本或元数据变化可能改变生成文件哈希；需同时核对专家内容和来源记录。

## 已验证硬件与启动配置

双路鲲鹏 920、512 GB RAM、单张 Ascend 910B4-1（64 GiB），4 NUMA 节点。机器的第二张同型号卡未执行本次模型。

CANN 9.0.0、Python 3.11.15、torch 2.10.0+cpu、torch_npu 2.10.0、transformers 5.12.1，驱动 25.5.1。镜像：`quay.io/ascend/sglang@sha256:5ed0acf504181fe092f9badbd8f77611b985ec65df0f93b66ab4b9616231f70b`。

安装依赖、编译 KT 并通过 NPU 算子 preflight 后，按固定教程配置 CANN 和 SGLang 环境，再设置：

```bash
export DSV4_MODEL_PATH=/path/to/DeepSeek-V4-Flash-W8A8-runtime-view
export DSV4_GGUF_DIR=/path/to/expert-gguf
export DSV4_GGUF_TEMPLATE="$DSV4_GGUF_DIR/dsv4_layer{layer_idx}_mxfp4.gguf"
export DSV4_NPU_DEVICE_ID=0
export DSV4_NUM_GPU_EXPERTS=32
export DSV4_CONTEXT_LENGTH=8192
export DSV4_CHUNKED_PREFILL_SIZE=512
export DSV4_THREADPOOL_COUNT=4
export DSV4_CPUINFER=96
export DSV4_MEM_FRACTION=0.81
export DSV4_PREFILL_STREAM=0
bash "$KTRANSFORMERS_REPO/kt-kernel/tools/ascend_dsv4/serve.sh" --foreground
```

本次 W8A8 的 `compress_ratios` 不压缩层标记需在单独 runtime config 中从 `1` 规范为 `0`，原文件保留。`tools/prepare_runtime_config.py` 提供路径无关的规范操作。

## 验证状态与限制

<!-- AUDIT_STATUS_BEGIN -->
**完整审计通过，已完成发布前验证。** 全部 43 层 × 256 专家 × 3 投影（33,024 个专家投影）通过编码/缩放字节比较及上游逐元素 bit-exact 验证；43 个 GGUF 的 SHA-256 已独立重读复核。审计后实际聊天测试仍正常。详见 `validation/full-audit.json`。
<!-- AUDIT_STATUS_END -->

模型加载、NPU decode 图捕获、`/health`、`/health_generate`、聊天及 SSE 流式均通过。中文首都问答返回“巴黎”，`17 + 25` 返回 `42`，单词指令返回 `READY` 并收到流式结束标记，见 `validation/runtime-validation.json`。

仅验证上述硬件/版本、单请求并发、8192 token 配置；没有持续吞吐、长上下文稳定性或完整准确率基准。原始文本补全曾出现重复文本，不据一条生成结果宣称质量完全等同官方实现。CPU 对 MXFP4 的软件处理不代表鲲鹏或 910B 有原生 FP4 矩阵计算能力。

不保证 CUDA、其他昇腾卡、其他 KT/SGLang 或通用 GGUF 推理器兼容；改变版本和启动参数后须重新验证。
