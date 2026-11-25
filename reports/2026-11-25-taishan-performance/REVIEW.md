# 双卡性能回读：DP-attention未改善单请求，长输入主要耗时在prefill

2026-11-25 回读继续轮 `performance-20261125-113036-profiler-resume`。五种配置的48条性能请求已完整核验，其中36条正式、12条预热。注意力TP2+MoE EP2的短负载decode约15.5 tok/s；注意力TP1/DP2+MoE EP2 graph的配置中位数约11.4 tok/s，TTFT仅略短。两种拓扑在1k/4k/7k输入上的prefill均约83–87 tok/s，7k首token等待约83秒。

这是性能测量和证据完整性验收，未声称完整模型质量或profiler验收。所有配置全局常驻64个专家、两卡各32个完整专家，CPU仅rank0计算64–255；96 CPU线程、4线程池，固定原W8A8/MXFP4权重。下文TP2/EP2表示注意力TP2、MoE EP2/TP1；DP-attention表示注意力TP1/DP2、MoE EP2/TP1。不是两套独立完整模型。

## 短输入：graph显著改善DP路径，但本轮总体仍慢于注意力TP2

实际输入144/792 token，固定输出512 token；每提示预热一次、正式三次，表为正式中位数。TTFT和TPOT来自原生输出token计数，而非SSE包数或首个非空正文块。

| 配置 | 输入token | TTFT（秒） | TPOT（毫秒/token） | decode（token/s） | 总耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| DP-attention eager | 144 | 2.447 | 369.39 | 2.707 | 191.205 |
| DP-attention graph | 144 | 2.142 | 87.95 | 11.369 | 47.087 |
| TP2/EP2 graph | 144 | 2.228 | 64.38 | 15.534 | 35.125 |
| DP-attention eager | 792 | 10.202 | 364.06 | 2.747 | 196.214 |
| DP-attention graph | 792 | 9.929 | 87.31 | 11.453 | 54.526 |
| TP2/EP2 graph | 792 | 10.151 | 64.63 | 15.472 | 43.180 |

DP graph相对TP2/EP2：TTFT短3.88%/2.19%，但TPOT长36.63%/35.08%、decode低26.81%/25.97%。对这组单请求负载，略小的首token差异没有补偿后续生成退化，不建议据此切换到DP-attention作为默认性能配置。

DP graph相对自身eager的decode中位数为4.20/4.17倍。这说明该实验路径下graph与eager的端到端表现差异很大；自由生成文本也随执行模式变化，因此不能将整个倍数都解释为纯图调度开销的收益。图路径有真实decode重放日志，不只是捕获成功。当前不建议继续用eager做长时间吞吐基准，除非它服务于具体诊断。

### 更正：792输入的DP0高速记录伴随正文异常

2026-11-25追加回查发现，DP0预热和正式请求虽然各报告512个输出token，最终正文都只有4个字符 `This`；DP1同负载的正文为2401个字符。此前仅称其“输出不同”不够准确。这两条应标为输出完整性异常，不能作为正常回答达到17 tok/s的证据。原始计数、中位数和记录均保留，不静默删除。

不能用合并中位数掩盖以下记录，也不能只取最快的一条当作配置速度：

| DP graph，792输入 | 请求身份 | decode（token/s） | TPOT（毫秒/token） | 输出 |
| --- | --- | ---: | ---: | --- |
| DP0 | 预热 | 16.851 | 59.34 | 512个token计数，正文仅为This |
| DP0 | 唯一正式重复 | 17.008 | 58.80 | 同上，SHA前缀86e1de7482 |
| DP1 | 正式重复1 | 11.418 | 87.58 | SHA前缀70e0948a11 |
| DP1 | 正式重复3 | 11.453 | 87.31 | 同上 |

每个提示的正式路由是DP1、DP0、DP1，因此原始配置中位数受DP1两条记录主导，含一条正文异常样本。144输入时两个rank都约11.2–11.4 tok/s，正文正常展开，并没有普遍的DP0高速现象。DP0的512个事件均逐个递增，整段四分之一窗口也均约17 tok/s，服务端日志和decode_throughput元数据与客户端一致；速度计数没有显示瞬时尖峰或漏计依据，异常在生成内容或输出链路。

请求固定ignore_eos=True以生成512 token，源码默认skip_special_tokens=True；大量EOS/特殊token被过滤是一个可能解释，输出链路问题也尚未排除。本轮采集器未保存原始输出token IDs，无法从现有记录确认实际生成了哪些特殊token。专家路由和计算负载可能随异常历史变化，不能直接把高速归因于NUMA或DP0调度优势。

后续先保存原始输出token IDs，并对照skip_special_tokens=False及自然EOS请求，区分特殊token生成与解码/输出问题；确认回答正常展开后，再平衡各rank重复次数、使用多条提示测性能。这个短输出诊断不需要先开展固定token全模型数值定位，本次没有启动新实验。

## 长输入：已跑通7168输入+64输出，prefill速度近似线性

实际每rank prefill块8192、SWA比例0.5、SWA池8192 token；其余模型权重与专家分工保持。下表每提示正式三次，输出固定64 token。graph只用于decode，streaming prefill关闭，prefill本身没有运行decode图。

| 输入token | 配置 | TTFT（秒） | 服务端prefill（秒） | prefill（token/s） | TPOT（毫秒/token） | decode（token/s） |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | TP2/EP2 | 12.394 | 12.340 | 82.99 | 66.48 | 15.042 |
| 1024 | DP-attention | 12.479 | 12.408 | 82.53 | 84.52 | 11.831 |
| 4096 | TP2/EP2 | 47.755 | 47.703 | 85.86 | 66.80 | 14.970 |
| 4096 | DP-attention | 47.707 | 47.641 | 85.98 | 88.18 | 11.341 |
| 7168 | TP2/EP2 | 83.015 | 82.964 | 86.40 | 67.01 | 14.922 |
| 7168 | DP-attention | 82.868 | 82.830 | 86.54 | 86.99 | 11.496 |

prefill吞吐差异仅为−0.55%、+0.13%、+0.16%，没有显示值得采用DP-attention的长提示收益。7k时约83秒总首token等待中，scheduler观察的prefill区间也约83秒；长提示优化优先级应高于继续微调这两种attention拓扑。服务器计时是forward_entry到prefill_finished的墙钟区间，含主机与完成开销，不是纯NPU核时间。

这批长输入使用不同提示、64输出和更大的块/池，不能将其与短输入512输出的decode速度差别全部解释为上下文增长。已经验证的是三种输入长度和每次64输出，不是8192上限、最大上下文、持续长输出或并发的验收。

## 资源与容量

以下为请求后的整卡HBM快照和整机MemTotal−MemAvailable，不是峰值分配、单模型独占RSS或CPU权重裁剪后的大小。

| 配置 | NPU2 HBM | NPU5 HBM | 系统内存压力参考 |
| --- | ---: | ---: | ---: |
| DP eager，短块 | 45.72 GiB | 45.67 GiB | 161.49 GiB |
| DP graph，短块 | 45.96 GiB | 45.90 GiB | 161.25 GiB |
| TP2/EP2 graph，短块 | 42.80 GiB | 42.72 GiB | 161.60 GiB |
| TP2/EP2 graph，宽prefill | 44.89 GiB | 44.76 GiB | 166.68 GiB |
| DP graph，宽prefill | 48.26 GiB | 48.00 GiB | 167.45 GiB |

DP完整注意力副本比对应TP2配置多占约3.2–3.4 GiB/卡，没有从本轮单请求性能中获得补偿。宽prefill相比各自短块额外使用约2.0–2.3 GiB/卡，完成7k输入时未OOM。这给出了一个实际容量点，但不能据此直接推算更大模型能放多少参数或最大峰值；CPU仍保存完整MXFP4集合，也没有新增模型部署。

## 验收、来源与限制

- 独立检查48条请求身份、冻结输入IDs、采样参数、DP路由、实际输入/输出数、SSE结束、正文SHA、全部逐token事件及计时；所有输出事件逐一递增，没有首包聚合或丢token。重新计算全部12组中位数及比较，与自动汇总一致。
- 每阶段86条专家驻留审计、DP完整注意力审计、两rank各1请求槽和实际SWA池大小通过。env/HCCL/实际DP合并probe通过，graph阶段有实际decode重放。非复用阶段测量到受控SIGTERM前没有scheduler异常或设备超时。
- 本机与远端81项已下载证据SHA一致；远端4个实际候选源码文件与candidate记录一致；6个本轮/原服务容器均已退出、PID0、restart=no、OOMKilled=false。
- 前两个阶段16条请求复用首轮完整测量。14项来源证据/复制前缀SHA一致，DP graph日志保留后续profiler失败，验收边界明确截在首次Profiling starts之前。其他三个阶段新跑，不覆盖首轮失败状态。
- 16条功能观察中算术均正确；DP的JSON格式仍失败，TP2/EP2通过。按用户性能优先规则记录，不阻断性能；这些题目不构成模型质量评测。
- 继续轮远端606条温度采样健康均OK、无采样错误，最高72°C/66°C；本机478条有效采样最高70°C/65°C，最低风扇5550 RPM，既有温控PID始终7468。首轮723条远端有效采样最高70°C/65°C，也没有触发实验温度条件。profiler没有再次启用。

各配置只有一次独立模型加载、每提示三次正式请求，按顺序运行且DP两个阶段跨中断复用；不提供有代表性的P95或普遍加速结论。不同配置的自由生成历史、路由可能不同。没有测试并发，不能否定DP对其他吞吐负载的潜在价值。

首轮profiler停止触发107020，异常trace仅保留用于诊断；本轮没有完成可用的算子性能时间线，因此CPU计算、搬运和HCCL占比仍未测出。不能把当前性能差异当作已定位的CPU或通信瓶颈。

## 下一步建议

优先建立匹配版本的streaming/layerwise prefill对照，先确认单卡既有路径是否能缩短1k/4k/7k的首token等待，再决定双卡适配。当前长提示瓶颈在可观察的prefill区间，继续改attention拓扑的收益不明显。

decode保持注意力TP2作为当日稳定对照，同时保留旧TP2/K32参考。下一项候选是CPU线程/NUMA或静态热专家放置，按单因素测TPOT；CPU等待和实际专家命中采用轻量计时/计数，不将失败的profiler自动停止路径重新插入正式测量。

DP0在792输入的高速记录值得用平衡rank、多提示短实验再确认，不直接取17 tok/s作达标值。扩大模型容量另立权重格式、CPU后端与峰值内存预算，本轮不自动下载或启动更大模型。

## 清理与复算

继续轮最终complete、cleanup_verified=true；实时回查两卡健康OK、均无计算进程、空载HBM约3409 MiB，18082已释放，原单卡服务保持停止。TP2/EP2两个容器受控停止后退出137、OOMKilled=false；请求和资源快照均在停止前完成，优雅退出仍是已知限制。

[独立核验程序](verify_results.py)、[逐次指标与分rank复核](runs/performance-20261125-113036-profiler-resume/reviewed-results.json)、[远端哈希与容器状态](runs/performance-20261125-113036-profiler-resume/remote-hash-check.json)、[原始请求](runs/performance-20261125-113036-profiler-resume/remote/requests.jsonl)、[自动汇总](runs/performance-20261125-113036-profiler-resume/remote/summary.json)。完整runs保留本机，按项目惯例不纳入Git。

```powershell
python reports/2026-11-25-taishan-performance/verify_results.py reports/2026-11-25-taishan-performance/runs/performance-20261125-113036-profiler-resume --remote-check
```

本次仅回读、验收与更新本地报告，没有启动新的后台实验、下载模型、修改温控或发布提交。
