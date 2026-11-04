# 公开权重与分享价值核查（2026-11-04）

结论：有针对 KT/Ascend 使用者的交付价值。公开社区已经有原始 MXFP4 权重和整模型 MXFP4 GGUF；本次检索及候选文件清单核查尚未找到与本候选包等价、可直接下载的 43 层 CPU 专家专用 GGUF。不能宣称全球没有、首发或唯一。

## 核查到的发布

| 仓库 | 文件清单核查 | 与本包的关系 |
| --- | --- | --- |
| [ggml-org / ModelScope](https://modelscope.cn/models/ggml-org/DeepSeek-V4-Flash-GGUF) | 12 项，8 个 GGUF；MXFP4 主模型为 2 个分片，另有 Q2 和 MTP 文件 | 整模型/分片发布，未见 `dsv4_layer*` 文件 |
| [bartowski / ModelScope](https://modelscope.cn/models/bartowski/DeepSeek-V4-Flash-GGUF) | 8 项，4 个 MXFP4 GGUF 分片 | 整模型/分片发布，未见 `dsv4_layer*` 文件 |
| [ggml-org / Hugging Face](https://huggingface.co/ggml-org/DeepSeek-V4-Flash-GGUF/tree/main) | 公开 MXFP4 模型与 llama.cpp 使用说明 | 证明公开 MXFP4 GGUF 已存在；不能把本包描述为首次提供 MXFP4 |
| [bartowski / Hugging Face](https://huggingface.co/bartowski/DeepSeek-V4-Flash-GGUF/tree/main) | 公开 MXFP4 模型 | 同上 |
| [6ms / Hugging Face](https://huggingface.co/6ms/DeepSeek-V4-Flash-MXFP4-GGUF/tree/main) | 4 项，2 个 GGUF，文件名分别为 MXFP4-F16-Q8Compat 和 MXFP4-FP8-BF16 | 未见 43 层专家拆分 |
| [yyj631 / Hugging Face](https://huggingface.co/yyj631/DeepSeek-V4-Flash-0731-KTransformers/tree/main) | API 返回提交 `a9ad07fd33e65172e62130f67419678b0b114f32`，仅 `.gitattributes`、`README.md` | 名称相关，但核查时无权重文件；且是 0731 命名，不能默认与本次原版相同 |
| [haanjack / Hugging Face](https://huggingface.co/haanjack/DeepSeek-V4-Flash-MXFP4/tree/main) | 53 项，无 GGUF | 名称包含 MXFP4 不等于所需 GGUF 格式 |
| [amd / Hugging Face](https://huggingface.co/amd/DeepSeek-V4-Flash-MXFP4/tree/main) | 75 项，无 GGUF | 同上 |

文件数为此次 ModelScope repo/files 或 Hugging Face API 的返回快照，目录项、元数据文件均可能计入，不代表权重分片数。仓库会继续更新。

[KT 官方 Ascend 教程](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md) 仍将 43 层 GGUF 列为单独工件，并提供本地转换步骤；已有相同 GGUF 集合时可以复用。教程支持该交付形式，但不能用来证明所有社区都没有预转换包。

## 本包可提供的价值

1. 对使用该 KT 昇腾路径的人，省去获取原始 159.63 GB 中间副本、自己运行转换及排查布局兼容的步骤。目标 CPU 文件仍有约 147.17 GB，不应夸大为大幅压缩或下载加速。
2. 逐文件 SHA-256、逐文件 immutable 源 revision、固定 KT/SGLang/llama.cpp 提交、全部专家验证证据形成可以核对的交付物。
3. 给出鲲鹏 920 + 单张 910B4-1 上实际通过的 W8A8 配套与图捕获配置，以及失败修复过程。

这属于工程打包与可复现验证贡献，没有引入新训练、新量化算法，也不意味着精度或速度优于已有发布。受众是对应 KT/Ascend 部署用户，不能将它宣传为通用 GGUF 整模型。

建议定位：**DeepSeek-V4-Flash KT CPU Experts / MXFP4 / Ascend-validated**。发布后可由用户决定是否在 KT 社区提交链接；本次没有代发帖子、issue 或外部消息。

## 检索边界

检查了网页关键词、HF 模型名检索和上述候选的实际文件列表；精确搜索过 `dsv4_layer0_mxfp4.gguf`、`dsv4_layer`、`per-layer MXFP4` 等。ModelScope 的全站搜索接口未成功枚举，只核查了已发现的候选仓库；未对所有模型仓库进行递归搜索。因此结论是“尚未找到等价包”，而非“社区绝对不存在”。
