# 一个 DeepSeek-V4-Flash 实例跨两张 910B：TP=2 短时功能验证通过

验收：2026-11-11（北京时间）。本轮 `20261111-kv-cap` 在独立 SGLang 副本中，以 TP=2、EP=1、KT CPU 专家卸载运行**一个模型实例、两个 TP rank、一个 API**，eager 与 decode 图模式的短请求均通过。它验证了本硬件及固定版本的这组运行条件，尚无性能提升、长上下文或系统准确率结论。当前已停止 TP 实验容器并恢复原单卡服务。

## 固定环境与必要变更

| 项目 | 本轮配置 |
| --- | --- |
| 硬件 | 双路鲲鹏 920、两张 64 GiB Ascend 910B4-1；物理设备 2/5 在容器内映射为逻辑设备 0/1；卡间拓扑为 `SYS` |
| 基线版本 | KT `c9873312643b615a717832946fba0346e69ab035`；SGLang `ff46d660d16cbf11ef393c80cc6b3ab7f6259780`；复用原服务 CANN 9.0.0 / torch_npu 2.10.0 环境 |
| 模型 | 现有 DeepSeek-V4-Flash W8A8 + 43 层 MXFP4 CPU 专家 GGUF，未更换权重或量化 |
| 模型并行 | `--tensor-parallel-size 2 --expert-parallel-size 1 --moe-a2a-backend none` |
| 专家 | 每层同一组 32 个 NPU 常驻专家，NPU 中间维度各分 1024；完整维度 2048；CPU wrapper 仅由 rank 0 创建 |
| CPU | `--kt-threadpool-count 4 --kt-cpuinfer 96`；保留单个 CPU 专家执行所有者 |
| 请求与缓存 | 单请求、上下文上限 8192、分块 prefill 512、`--max-total-tokens 16384`；未验证完整 8192-token 请求 |
| 模式 | 先 eager（`--disable-cuda-graph`），再独立启动图模式；本轮只验证 batch=1 的 decode 图捕获，prefill 图未启用 |
| 隔离 | 由原容器创建本地环境快照镜像，基线源码/权重只读复用；候选源码和日志在独立 `/experiment` |

当前固定 SGLang 原本显式拒绝 NPU + KT 的 TP≠1。候选 [candidate.diff](candidate.diff) 仅在 `KT_NPU_EXPERIMENTAL_TP2=1` 时允许 TP=2/EP=1，并输出逐层分片与 CPU 所有者诊断。它利用已有 rank 0 CPU 执行及 NPU 分片/求和路径，没有新增通用 TP 算法，也没有证明其他 TP/EP 配置可用。基线源码未修改。候选源码身份和环境镜像 ID 见[验收摘要](acceptance-summary.json)。

## 验收证据

| 项目 | 结果与证据 |
| --- | --- |
| 两卡 HCCL | 两 rank 对 1、1024、262144 个 FP32 元素求和均精确得到 3；仅小规模功能探针，非通信性能基准 |
| eager 两 rank | 两个 rank 各有 43 层诊断，局部/完整中间维度 1024/2048；rank 0 `cpu_owner=True`，rank 1 `False`；两卡各有 scheduler 进程 |
| 缓存上限 | 两 rank 日志均显示最终 `full=16384, swa=1536, c4=4096, c128=128`；自动预算仍先打印约 376 万，但随后按上限重新计算，实际分配未 OOM |
| eager 短请求 | `巴黎`、`42`、`READY` 三项通过，均有 usage、`finish_reason=stop`、SSE `[DONE]` |
| decode 图捕获 | 两 rank 均有 `Capture target decode NPU graph end`，分别约 6.64/6.72 秒；capture batch 为 1 |
| 图模式短请求 | 同三项全部通过；正常执行后由控制器主动停止 |
| 原服务恢复 | 原单卡容器重启并通过同三项短请求；流程状态 `complete`、`original_restored=true` |

eager 在 11:55:40 进入验收，图模式在 12:00:13 进入验收，12:00:50 完成两个模式的短测，12:11:34 完成原服务恢复。以上为远端北京时间；原始容器日志使用 UTC，本机控制器时钟与服务器约有 30 秒偏差，未将不同主机的时钟混算耗时。

eager 时两卡 HBM 快照为 26,956/26,898 MiB；图模式为 27,204/27,147 MiB，约每卡 26.5 GiB。这是 NPU 设备总 HBM 使用量，含运行时和缓存；它既不是精确权重大小，也不能直接与此前没有缓存上限的单卡配置解释为纯 TP 节省比例。

## 短请求耗时及解释边界

| 请求 | 实际输入 / 输出 token | eager TTFT / 总时长 | 图模式 TTFT / 总时长 |
| --- | ---: | ---: | ---: |
| 法国首都 | 14 / 2 | 1.043 / 1.426 s | 0.756 / 0.815 s |
| `17 + 25` | 16 / 2 | 0.994 / 1.334 s | 0.753 / 0.826 s |
| `READY` | 12 / 3 | 0.953 / 1.661 s | 0.700 / 0.844 s |

各模式每条只有一次，输出只有 2–3 token，且跨独立进程启动。它们只用于功能验收，不能换算稳态 tokens/s，不能据此认定双卡比单卡快或给出图模式加速比。三个极短确定性输出不足以证明 TP 的数值等价或完整模型质量，仍需固定历史的数值检查及代表性任务验证。

## 热保护、退出与最终状态

本机 154 个有效热采样，两卡最高均 52°C，四风扇最低 7350 RPM。另有三次独立 SSH 命令超时，连续缺测计数均为 1，未触发连续两次失败中止；没有温度、健康或风扇告警。

两个实验容器在 SIGTERM 主动结束后均打印了 `sigterm_watchdog` → `SystemExit: 0` 和 asyncio `CancelledError` 堆栈，最终容器退出码为 137。异常位于已完成短请求之后的退出阶段，未观察到服务期间同类异常；本轮因此只能确认功能门槛与最终清理，不能把关停写成优雅退出已验证。退出信号处理仍有待单独排查。

14:35 再次只读核查：仅原单卡容器在运行，`192.168.137.11:18080/health` 返回 200；TP 的 HCCL/eager/graph 容器均已停止，18082 不再监听，NPU 5 无模型进程，两卡均 `OK`、53°C。本轮为试运行并已恢复基线，当前 API 仍由原单卡配置提供。

## 归档与复核

完整本机数据：`runs/20261111-kv-cap/remote/`；远端：`/data/deploy/dsv4-tp2-20261111-kv-cap`。`runs/` 被 Git 忽略，日志与源码未删除。可提交的小体积[验收摘要](acceptance-summary.json)保留全部九条短请求结果、HCCL 结果、源身份、热统计和证据 SHA-256。复核入口：

```powershell
python reports/2026-11-10-taishan-tp2/verify_results.py --run reports/2026-11-10-taishan-tp2/runs/20261111-kv-cap --output reports/2026-11-10-taishan-tp2/acceptance-summary.json
```

前两轮失败分别见[停止竞态](REVIEW_FIRST.md)和[缓存自动预算 OOM](REVIEW_KV_OOM.md)。下一项有价值的验证是同配置缓存上限下的 TP=1/TP=2 长生成对照及数值/任务门槛，再决定是否增加 NPU 常驻专家。当前没有以本轮短测结果修改生产默认或向上游声明通用支持，也没有自动启动新的性能实验。
