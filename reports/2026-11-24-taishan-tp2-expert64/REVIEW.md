# DS-V4-Flash TP2/K64：部署通过，decode 未因专家扩容加速

2026-11-24 回读结论：**TP2/K64 的 eager、decode graph、实际专家驻留及短任务均已跑通。与 TP2/K32 相比，首 token 时间缩短约 7.5%–9.3%，但客户端生成速度降低约 1.2%–1.5%，每张卡额外占用约 16.16 GiB HBM。** 本次未证明扩大常驻专家集合能改善连续生成速度。

结果可用于本次短请求的探索性比较。完整数值一致性、独立任务质量、并发和长上下文仍未验收；原始 `correctness_accepted=false` 保留。

## 性能结果

以下均为三次正式请求的中位数，单请求、实际输入 144/792 token、固定生成 512 token；预热各一次，不计入正式统计。K 表示每层驻留 NPU 的逻辑专家集合总数，TP2 两个 rank 对同一集合做中间维度分片。

| 配置 | 输入 token | TTFT（秒） | 客户端生成速度（token/s） | 整请求时间（秒） |
| --- | ---: | ---: | ---: | ---: |
| TP1/K32 | 144 | 2.461 | 15.041 | 36.429 |
| TP2/K32 | 144 | 2.434 | 15.948 | 34.476 |
| TP2/K64 | 144 | 2.252 | 15.760 | 34.663 |
| TP1/K32 | 792 | 11.182 | 15.021 | 45.262 |
| TP2/K32 | 792 | 11.227 | 15.827 | 43.523 |
| TP2/K64 | 792 | 10.182 | 15.582 | 42.921 |

| K64 相对的基线 | 输入 token | TTFT 缩短 | 生成速度变化 | 整请求耗时缩短 |
| --- | ---: | ---: | ---: | ---: |
| TP1/K32 | 144 | 8.48% | +4.78% | 4.85% |
| TP1/K32 | 792 | 8.94% | +3.73% | 5.17% |
| TP2/K32 | 144 | 7.48% | −1.18% | −0.54%（稍慢） |
| TP2/K32 | 792 | 9.31% | −1.55% | 1.38% |

因此，K64 相比单卡旧基线有小幅改善；**在已经使用 TP2 的条件下，把 K32 增至 K64 的整请求收益很小且随输入长度变化**。TTFT 包含 prefill、首个非空正文块与客户端观测开销，不是服务端纯 prefill 时间。当前没有分项 profiler，不能把 TTFT 变化或 decode 退化直接归因为 CPU、NPU 算子或通信中的某一项。

## 驻留与资源

| 配置 | 每层 NPU / CPU 专家数 | 请求后的 NPU 2 HBM | 请求后的 NPU 5 HBM | 系统 MemTotal−MemAvailable |
| --- | ---: | ---: | ---: | ---: |
| TP1/K32 | 32 / 224 | 46.58 GiB | 3.33 GiB（空载） | 153.47 GiB |
| TP2/K32 | 32 / 224 | 26.67 GiB | 26.60 GiB | 160.51 GiB |
| TP2/K64 | 64 / 192 | 42.83 GiB | 42.76 GiB | 160.39 GiB |

HBM 是整卡 `npu-smi` 观测值，包含模型及运行开销；最后一列是系统内存压力参考值，不是模型独占 RSS。两个 TP2 配置没有观察到明显的 RAM 释放，不能把少 32 个 CPU 计算专家解释为 RAM 等比例减少。

K64 的全部 43 层、两个 rank 共 86 条驻留审计均通过：mask 是前 64 个专家，实际 `w13_weight` / `w2_weight` 专家轴均为 64，CPU executor 只存在于 rank 0。NPU 中间维度 local=1024、full=2048；第 0 层实际张量形状为 `w13=[64,4096,2048]`、`w2=[64,1024,4096]`。相对 K32 的每卡约 16.16 GiB HBM 增量与新增常驻专家相符。

## 功能与证据核验

- 新轮环境探针及两个 rank 的 HCCL 三种规模求和通过。实际 kt-kernel 扩展 SHA-256 为 `9b21e209ff3cfd63fe1068b18516b686962f69b7f4f8342a4bcef366931ef8c8`。
- TP2/K64 eager 三道短题、三个图配置各五道短题均正确且自然停止，SSE 结束完整。两个 TP2 配置均有双 rank decode 图捕获成功和实际重放日志。
- 正式矩阵为 18 条请求，加 6 条预热；逐请求提示、seed、512 token、SSE 正文、结束标记、哈希和计时完整。独立使用原始 chunk 时间戳重算速度、中位数和比值，与自动汇总一致。
- 新轮本机与远端 61 项证据 SHA 一致，前驱来源 21 项 SHA 一致，冻结 runtime 源码 SHA 一致。复用日志保持原字节，TP1 八条请求作为前缀保留于新请求文件，没有重复计数。
- 新轮 311 条远端有效温度采样，健康均 OK、无采集错误，最高 70°C / 65°C；本机现有温控服务 PID 始终为 7468，风扇最低 5550 RPM。90°C / 10 秒条件没有触发，新轮也没有记录 SSH 传输错误。

## 实验范围与复用来源

使用已在硬件验证过的 Pan-Boyi SGLang `ff46d660d16cbf11ef393c80cc6b3ab7f6259780`、原 TP2 opt-in 补丁与启动期审计，匹配 KT `c9873312643b615a717832946fba0346e69ab035` 归档及原 W8A8/MXFP4 权重；不是官方 main 的新硬件验证。环境镜像固定为 `sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e`。

96 CPU 线程、4 线程池、EP=1、单请求，mem-fraction=0.81、chunked-prefill=512、max-total-tokens=16384；streaming prefill、dynamic resident、side stream 和 radix cache 关闭。配置的 context-length=8192 不代表 8192 token 请求已经验证。

最终运行是 `expert64-20261124-ssh-resume`。TP1/K32 六条正式请求及 K64 eager 来自 `expert64-20261124-cooling90` 的完整阶段，图模式 TP2/K32、TP2/K64 在新轮执行。TP1 约在服务端 10:31–10:37 测量，两个 TP2 约在 11:27–11:43 测量；这是**跨中断的顺序比较，每配置只有一次独立模型启动**，不能作为交错 A/B 或普遍加速承诺。

同配置三次正式输出哈希一致，不同 TP/K 配置之间的生成文本不同。性能仅比较相同提示与输出长度，decode 历史与专家路由没有固定；K32→K64 还会把额外专家从 CPU MXFP4 路径改为 NPU W8A8 路径。短题通过不能关闭此前 TP1/TP2 数值差异或独立任务质量的缺口。

## 清理状态

后台已完成，新建及复用来源模型容器均为 exited/PID 0、restart=no、OOMKilled=false，无 NPU 计算进程，18082 已释放，原单卡服务保持最初的停止状态。本轮回读为两卡健康 OK、空载 HBM 约 3406/3407 MiB；没有再次启动模型或修改既有温控服务。

模型容器受控停止后退出码仍为 137，并在 SIGTERM 后出现 `SystemExit: 0` / `CancelledError` 堆栈；所有请求及资源快照在停止前完成。清理已通过，但优雅退出仍是已知限制，不能将退出码 137 当作 OOM，也不能声称优雅停止已经验收。

## 可复算记录

- [独立核验程序](verify_results.py)
- [复核结果、逐次指标与限制](runs/expert64-20261124-ssh-resume/reviewed-results.json)
- [本机/远端证据哈希及容器状态](runs/expert64-20261124-ssh-resume/remote-hash-check.json)
- [原始请求](runs/expert64-20261124-ssh-resume/remote/requests.jsonl)、[自动汇总](runs/expert64-20261124-ssh-resume/remote/summary.json)、[复用来源](runs/expert64-20261124-ssh-resume/remote/checkpoint-reuse.json)

```powershell
python reports/2026-11-24-taishan-tp2-expert64/verify_results.py reports/2026-11-24-taishan-tp2-expert64/runs/expert64-20261124-ssh-resume
```

本次验收仅覆盖既定部署、短功能、驻留、数据完整性和探索性性能比较，不自动开始新一轮优化或发布社区评论。
