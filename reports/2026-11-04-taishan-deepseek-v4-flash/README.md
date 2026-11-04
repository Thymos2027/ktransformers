# taishan 部署 DeepSeek-V4-Flash：下载、传输与验收

本次部署于 **2026-11-04 08:16（北京时间）通过服务器验收**，08:36 通过 Windows 客户端的局域网复验。采用 KTransformers CPU 专家卸载与 SGLang 昇腾推理，在一张 Ascend 910B4-1 上提供 OpenAI 兼容接口。

| 项目 | 实际结果 |
| --- | --- |
| API Base URL | `http://192.168.137.11:18080/v1` |
| 模型 ID | `/workspace/models/DeepSeek-V4-Flash-W8A8` |
| 容器 | `dsv4-kt-20261104` |
| 服务器工作目录 | `/data/deploy/dsv4-20261104` |
| 验收状态 | `serving_verified`；首个配置成功，未使用后备配置 |
| 并发与上下文 | 单请求并发，8192 token 上下文，512 token 分块 prefill |
| 专家执行 | 每层 32 个常驻专家在 NPU，另外 224 个专家在 CPU |
| NPU 图捕获 | 已启用并通过验收，非 eager 后备路径 |

这是功能部署验收记录，尚未做持续吞吐、并发、长上下文稳定性和系统准确率基准。原始 `/generate` 补全文本出现了重复片段，故不能仅凭该测试包含 Paris 就认定模型质量全面达标；结构化聊天接口另外验证了明确答案和流式结束。

## 1. 硬件与执行方式

| 项目 | 本次机器配置 |
| --- | --- |
| 服务器 | 华为 TaiShan 200（2280），SSH 别名 `taishan`，LAN IP `192.168.137.11` |
| CPU | 双路鲲鹏 920 7260，合计 128 核，4 个 NUMA 节点 |
| 内存 | 512 GB 物理内存，系统可见约 502 GiB |
| NPU | 2 × Ascend 910B4-1，每张 64 GiB；本次使用其中一张 |
| 设备映射 | 物理 NPU 2 / `/dev/davinci2` 对应逻辑设备 0；物理 NPU 5 对应逻辑设备 1 |
| 系统 | Ubuntu 24.04.4，aarch64；驱动 25.5.1 |
| 模型磁盘 | `/data`，约 65.5 TiB；部署时可用约 60 TiB |

910B4-1 不提供原生 FP4/MXFP4 或 FP8 矩阵计算路径。鲲鹏 920 同样没有原生 MXFP4 指令；KT 的 CPU 算子处理压缩数据、解码和缩放，再使用 CPU 支持的运算完成计算。CPU 能处理 MXFP4 指的是软件算子支持，不能据此推断有原生 FP4 加速。

本次 NPU 端使用 **W8A8 INT8** 权重/激活计算路径，CPU 端使用逐层 MXFP4 GGUF。原始权重用于提取和重排 CPU 专家，并不是直接整套加载到 NPU。SGLang 负责模型执行、调度、注意力、NPU 图与 HTTP 接口；KT 提供 CPU 专家算子和卸载支持。仓库中的 `ascendc_mxfp4` 代码是 MXFP4 到 W8A8 的软件转换实现，不能据其名称声称 910B 原生支持 FP4。

依据：[华为 CANN 量化格式适用范围](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/amct/atlasamct_16_0049.html)、[Ascend C 产品数据类型表](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlas_ascendc_10_0019.html)、[本仓库昇腾部署教程](../../doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md)。

## 2. 模型来源、格式与大小

| 文件集 | 固定模型来源 | 文件数 | 大小（十进制） | 用途 |
| --- | --- | ---: | ---: | --- |
| 原始权重 | [deepseek-ai/DeepSeek-V4-Flash，ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) | 74，含 46 个权重分片 | 159,630,045,338 字节，约 159.63 GB | 生成 CPU 专家 GGUF |
| W8A8 | [sgl-npu/DeepSeek-V4-Flash-W8A8，ModelScope](https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8) | 65，含 46 个权重分片 | 294,635,333,768 字节，约 294.64 GB | NPU 注意力、稠密层及常驻专家 |
| 派生 GGUF | 从上述原始权重提取与重排 | 43 个逐层文件 | 每文件 3,422,552,640 字节，合计约 147.17 GB | KT `LLAMAFILE` CPU 专家路径 |

原始模型已经采用混合精度压缩：专家主要是 MXFP4，两个 4-bit 编码打包成一个字节，并携带分组缩放；其他部分还有 FP8、BF16 等。MXFP4 专家加上每 32 个元素的 8-bit 缩放，约为 4.25 bit/权重。因此原始文件比 W8A8 小，不能把“原始权重”理解为整套 BF16。

本次选定的是 `compressed-tensors` W8A8。Modelers / Eco-Tech 的其他 W8A8 发布未确认与本次格式和哈希相同，未替换当前文件集。同名 Hugging Face 仓库经代理返回 401，直连超时，未作为本次 W8A8 下载来源。没有把不同发布者、不同版本或其他 GGUF 的同名文件直接接入断点。

完整逐文件大小、SHA-256 和本机下载完成时间保存在 [原始权重校验清单](artifacts/native-download-verification.json) 与 [W8A8 校验清单](artifacts/w8a8-download-verification.json)。

## 3. 下载与网络排障

本机保存目录：

```text
D:\Models\DeepSeek\DeepSeek-V4-Flash
D:\Models\DeepSeek\DeepSeek-V4-Flash-W8A8
D:\Models\DeepSeek\_download-20261031
```

下载器按冻结的 ModelScope 文件 revision、大小、SHA-256 下载。采用 `.part`、HTTP Range 续传，完成后逐文件 SHA-256 正确才改为正式文件名。错误正式文件保留并报错，错误临时文件隔离；文件锁防止重复工作进程。

最终有效配置为 **ModelScope 直连、强制 IPv4、8 个文件并发、API repo 入口**，见 [网络策略快照](artifacts/download-network-policy.json)。Requests 禁用环境与 Windows 注册表代理查找；没有因测试而修改全局系统代理。

| 问题或尝试 | 观察与处理 |
| --- | --- |
| ModelScope 500、读超时、连接中断 | 保留断点并重试，后台监督器对网络类失败延迟重启 |
| 本地 7890 代理 | 短测偶有提升，后续复测变慢或超时；未采用为最终下载线路。经过 7890 也不能直接认定流量走了境外节点 |
| API / resolve 两种入口 | 均可返回 Range，最终仍到同一个 CDN，不能当作两个独立模型源 |
| IPv4、IPv6 与 CDN 节点对照 | IPv4 + 8 并发曾实测模型接收约 9.81 MiB/s，占网卡接收约 96%；换网后也曾降到 0.35 MiB/s 以下，不能把短测外推为稳定带宽 |
| 网卡 150 Mbps 协商速率 | 只是链路协商值，不能直接用作模型下载 ETA |
| 低速触发累计 100 次重试上限 | 2026-11-03 14:11 停止；改为有进展时不因累计次数停止，仍保留连续失败上限 |
| 过于频繁的低速重连 | 默认判定从 90 秒 / 128 KiB/s 调整为 300 秒 / 16 KiB/s，避免慢速但仍在前进的连接反复重建 |
| 监督器未识别低速失败 | 补充 `Sustained speed below`、`IncompleteRead`、`Connection broken` 分类；完整性失败不当作网络恢复处理 |

续传和重试修复通过了本地测试，包括从非零偏移续传、最终哈希、已完成文件复用、错误文件保留、取消后不启动新文件，以及连续有进展超过 100 次尝试后仍正确完成。

### 下载起止时间与总用时

以下时间均为 **北京时间（UTC+8，2026 年）**，依据 [下载时间线日志摘录](artifacts/download-timeline.log) 及两套权重的下载校验清单。

**整个下载流程于 10 月 31 日 17:35:32 启动，至 11 月 3 日 23:13:52 全部下载并校验完成，总历时 3 天 5 小时 38 分 20 秒（77 小时 38 分 20 秒）。** 起点是首次获取固定文件清单，随后下载配置/索引并做真实权重样本验证；17:37:24 进入全量下载阶段。

| 阶段 | 开始时间 | 下载及校验完成时间 | 阶段历时 |
| --- | --- | --- | --- |
| 原始权重全量下载 | 10 月 31 日 17:37:24 | 11 月 2 日 02:56:31 | 1 天 9 小时 19 分 7 秒 |
| W8A8 全量下载 | 11 月 2 日 02:56:31 | 11 月 3 日 23:13:52 | 1 天 20 小时 17 分 21 秒 |
| 两套权重全量下载合计 | 10 月 31 日 17:37:24 | 11 月 3 日 23:13:52 | 3 天 5 小时 36 分 28 秒 |

两套权重的全量下载按顺序进行；W8A8 的配置和索引已在 10 月 31 日的前置验证阶段提前下载。以上均为实际经过的时间，包含暂停、断线、重试、线路排障和本机 SHA-256 校验，不能视为持续满速传输用时。`status.json` 的 `started_at` 会随下载器重启更新（最终一次为 11 月 3 日 14:27:32），不代表整个任务首次启动时间。总计下载约 **454.27 GB**，不含派生 GGUF，也不包含后续上传服务器、转换与部署的用时。

## 4. 到 taishan 的传输

目标分别是 `/data/llm/DeepSeek-V4-Flash` 与 `/data/llm/DeepSeek-V4-Flash-W8A8`。只创建和使用任务自己的子目录，未修改 `/data/llm` 原有的父目录属主。

采用 Windows 原生 OpenSSH SFTP，2 文件并发；新文件用 `put`，已有非零临时文件用 `reput`。上传写入 `.transfer.part`，服务器 SHA-256 正确后才改成正式名称。只传本机已经完成校验的文件，下载中的 `.part` 不传。

| 批次 | 文件数 | 字节数 | 完成时间（北京时间） |
| --- | ---: | ---: | --- |
| 原始权重 | 74 | 159,630,045,338 | 11 月 2 日 15:45 |
| W8A8 第 1 批 | 47（31 个权重分片） | 200,470,758,987 | 11 月 3 日 16:08 |
| W8A8 第 2 批 | 12（9 个权重分片及索引、tokenizer 文件） | 54,253,048,137 | 11 月 3 日 19:01 |
| W8A8 第 3 批 | 6（剩余权重分片） | 39,911,526,644 | 11 月 4 日 00:05 |

曾修复两处传输问题：`reput` 无法创建不存在的远端文件，改为按偏移选择 `put/reput`；后台 SSH 请求挂起，改为 `ssh -n` 并把标准输入设为 `DEVNULL`。

最后一批完成后核对了完整的 65 文件清单。逐批传输时已经读取并验证远端 SHA-256；全套收口时复用路径、大小、哈希、mtime 均一致的 65 份记录，没有再次读取整套 294.64 GB。依据见 [完整远端验证结果](artifacts/w8a8-remote-verification.json)，不能表述为最后又完整重算了一遍 SHA-256。

三批状态：[第 1 批](artifacts/transfer-batch1-status.json)、[第 2 批](artifacts/transfer-batch2-status.json)、[第 3 批及全套收口](artifacts/transfer-batch3-status.json)。

## 5. 固定版本与部署参数

| 组件 | 固定版本或来源 |
| --- | --- |
| KTransformers | `c9873312643b615a717832946fba0346e69ab035`，本项目提交的干净归档 |
| SGLang | `Pan-Boyi/sglang`，分支 `dsv4-cann9-no-patch`，提交 `ff46d660d16cbf11ef393c80cc6b3ab7f6259780` |
| llama.cpp | `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`，构建时应用 KT 自带补丁 |
| pybind11 | `bb05e0810b87e74709d9f4c4545f1f57a1b386f5` |
| 容器镜像 | `quay.io/ascend/sglang@sha256:5ed0acf504181fe092f9badbd8f77611b985ec65df0f93b66ab4b9616231f70b`，ARM64 |
| 容器运行栈 | CANN 9.0.0、Python 3.11.15、torch 2.10.0+cpu、torch_npu 2.10.0、transformers 5.12.1 |
| 主机 CANN | 原先为 8.5.0；本次未替换主机 CANN 或驱动，使用容器内工具链 |

镜像已带 `sgl_kernel_npu`、`deep_ep`、`attentions`、`custom_ops` 及 `customize/custom_transformer` vendors；KT Ascend 扩展在本次单独构建。安装 Python 依赖时约束镜像中现有 torch 家族版本，KT wheel 使用 `--no-deps` 安装。

原始与 W8A8 权重通过 `/data/llm:/models:ro` 只读挂载。运行视图 `/workspace/models/DeepSeek-V4-Flash-W8A8` 链接原文件，单独生成 config，将 `compress_ratios` 中不压缩层的标记 `1` 转为分支所需的 `0`；未改动下载目录中的原始 config。

| 配置 | 实际值 |
| --- | --- |
| `--device` / `--attention-backend` | `npu` / `ascend` |
| `--quantization` / `--dtype` | `compressed-tensors` / `bfloat16` |
| TP / EP | 1 / 1 |
| `--kt-method` | `LLAMAFILE` |
| `--kt-num-gpu-experts` | 32 |
| `--kt-threadpool-count` / `--kt-cpuinfer` | 4 / 96 |
| `--context-length` / `--chunked-prefill-size` | 8192 / 512 |
| `--max-running-requests` | 1 |
| `--mem-fraction-static` / `--page-size` | 0.81 / 128 |
| GGUF 模板 | `/workspace/gguf/dsv4_layer{layer_idx}_mxfp4.gguf` |
| 服务绑定 | 容器内 `0.0.0.0:18080`，Docker 映射至主机 `192.168.137.11:18080` |

## 6. 部署过程、排障与校验范围

后台顺序：源权重记录与大小检查 → 容器系统依赖 → 固定源码 → Python 依赖 → KT 编译 → 复用镜像 NPU 算子 → GGUF 转换与校验 → preflight → 启动 → 健康、生成、聊天测试。失败记录阶段和日志；成功后持续运行模型进程。

第一次流程于 **00:14** 失败：KT 源码归档中的 Shell 文件带 CRLF，Linux 报 `$'\r': command not found`。**07:52** 恢复，在服务器工作副本中把 Shell、补丁与 CMake 文件的 CRLF 规范为 LF，共处理 38 个文件；修正 Python 可执行路径为 `command -v python3`，已有系统依赖不重复安装。变更清单见 [换行规范记录](artifacts/source-line-ending-normalization.json)。

随后于 **08:05** 完成 GGUF 转换检查，**08:06** 开始加载，**08:16** 通过验收。预设的 16 个常驻专家和 eager 后备配置均未使用。

GGUF 的实际校验范围：

- L1：43/43 个逐层文件存在，全部大小一致且精确匹配。
- L2：未提供 SHA-256 self-manifest，上游脚本明确跳过；**没有完成派生 GGUF 的整套 SHA-256 自清单审计**。
- L3：第 0、14、28 层完成与原始专家数据的逐元素 bit-exact 检查。

原始/W8A8 源权重的逐文件 SHA-256 与上述派生 GGUF 检查是两组不同证据。不能把源文件已经校验，等同为对全部 GGUF 都做过全量 bit-exact 或哈希审计。

## 7. 验收结果

服务器验收详见 [原始结果 JSON](artifacts/server-acceptance.json)、[验收日志](artifacts/server-acceptance.log)、[preflight](artifacts/preflight.log)。

| 检查 | 结果 |
| --- | --- |
| KT Ascend 扩展 | `init_ascend_callback_worker` 存在，MXFP4 类型 ID 为 39 |
| NPU 自定义算子与源码导入 | 通过，SGLang 指向固定源码工作副本 |
| `/health`、`/health_generate` | HTTP 200 |
| 图捕获 | `Capture target decode NPU graph end`，通过 |
| `/generate` 原始补全 | 包含 Paris；也有重复文本，保留完整结果供复查 |
| 聊天 API | 单词指令返回 `READY` |

本机使用 Requests `trust_env=False`，从 LAN 直接访问接口，复验结果见 [客户端验收 JSON](artifacts/client-acceptance.json)：

| 测试 | 输出 | 本次请求耗时 |
| --- | --- | ---: |
| 中文问法国首都 | `巴黎` | 2.375 秒 |
| `17 + 25` | `42` | 1.531 秒 |
| SSE 流式单词回复 | `READY` 且收到 `[DONE]` | 1.265 秒，首次内容约 1.062 秒 |

这些是单次短请求耗时，不是 tokens/s 基准。验收时物理 NPU 2 的 HBM 使用约 57,546 MiB；物理 NPU 5 没有模型进程。主机已用内存约 153 GiB，但存在缓存和 swap，不能据此当作模型精确常驻内存。

## 8. 使用、检查和后续工作

通过 OpenAI 兼容客户端填写本报告开头的 Base URL 与模型 ID。局域网接口本次未配置 API key，仅适用于现有可达网络；跨网络访问和认证配置不在本次验收范围。

```powershell
ssh taishan 'cat /data/deploy/dsv4-20261104/status.json'
ssh taishan 'sudo -n docker logs --tail 60 dsv4-kt-20261104'
ssh taishan 'cat /data/deploy/dsv4-20261104/acceptance.json'
```

状态 JSON 是阶段快照，服务存活需同时查看容器状态或实时请求 `/health`。容器名是 `dsv4-kt-20261104`，状态与日志不在先前的模型下载/传输目录。

```python
import requests

session = requests.Session()
session.trust_env = False
base = "http://192.168.137.11:18080/v1"
model = session.get(f"{base}/models", timeout=15).json()["data"][0]["id"]
response = session.post(
    f"{base}/chat/completions",
    json={
        "model": model,
        "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
        "temperature": 0.6,
        "max_tokens": 128,
    },
    timeout=300,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

停止本次服务：`ssh taishan 'sudo -n docker stop -t 90 dsv4-kt-20261104'`。本次没有配置系统级自启动或健康故障自动重启；直接 `docker start` 会重新运行部署 pipeline，不能当作只启动模型的快捷命令。

依赖拉取阶段使用 Windows 本机 LAN 代理 `192.168.137.1:7890`；已部署的推理使用本地文件，不再依赖该代理下载模型。

后续可独立进行持续性能基准、CPU/NUMA 亲和性与线程数调优、第二张 NPU 的利用、长上下文和并发验证、GGUF 全量审计，以及拆分部署/服务启动和持久化运行管理。当前未将转换后的权重发布到社区。

## 9. 归档与复现

`scripts/` 保存实际部署脚本、环境配置和客户端测试；`artifacts/` 保存验收结果及下载/传输清单；[SHA256SUMS](SHA256SUMS) 用于核对本报告归档文件。

脚本含本机/服务器固定路径、镜像、IP 和代理，属于本次部署快照，不能直接当作其他机器的通用安装器。`pipeline.sh` 依赖 `ktransformers-c987331.tar.gz`，此约 66.8 MB 的源码归档未纳入报告，可从对应提交重新生成：

```powershell
git archive --format=tar.gz --output=ktransformers-c987331.tar.gz c9873312643b615a717832946fba0346e69ab035
```

本次实际归档 SHA-256：`04ece47b6c417d31943674ce5b8663d02c54c2a2b2b733aaf80b9590624fe5b6`。重新生成的压缩包未必字节相同，应检查提交与内容；原归档的换行规范步骤已包含在部署脚本中。

更完整的下载与部署现场仍在本机 `D:\Models\DeepSeek\_download-20261031`、`_deploy-taishan-20261104`，以及服务器 `/data/deploy/dsv4-20261104`。4 个已完成的本机 `_transfer-taishan-*` 目录已归档为 [传输现场压缩包](artifacts/transfer-workspaces.zip)，待清理；下载器的 6 个 `repair-test-*` 临时目录和 `__pycache__` 缓存也可清理。本次删除操作被执行策略拦截，以上目录尚未删除。模型权重目录保留。本项目报告未复制权重、GGUF、容器层或全部历史日志。
