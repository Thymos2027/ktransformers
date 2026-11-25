# TaiShan 双卡：性能测量与 prefill 容量扩展

2026-11-25（北京时间）。用户要求按性能与有限资源运行更大模型的方向推进实验。本轮优先补齐注意力TP1/DP2+MoE EP2的TTFT/TPOT、eager/graph同负载对照，及较长输入下的prefill速度。固定token数值定位不作为性能实验前置条件；JSON格式失败单独记录。没有启动新模型下载或替换权重。

修订：首轮已完成DP eager/graph的16条请求，随后请求内profiler自动停止触发NPU同步超时，见[首轮复盘](REVIEW_FIRST.md)。继续轮关闭正式测量进程中的profiling API，短trace暂缓；严格校验源码/环境/权重/输入后复用16条记录，继续其余32条请求。原始首轮失败状态不变，最终汇总明确跨运行复用和trace未完成。

## 固定环境与五次模型加载

SGLang基础 `ff46d660d16cbf11ef393c80cc6b3ab7f6259780`、已有Pan-Boyi fork和既有EP2/DP HCCL修复；镜像 `sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e`；匹配KT `c9873312643b615a717832946fba0346e69ab035`归档、原W8A8/MXFP4文件。只改隔离SGLang副本，统一候选支持两种attention拓扑；86条专家驻留审计、完整注意力审计、HCCL实际合并probe均保留。环境不升级，温控服务不改。

| 阶段 | Attention | MoE | 执行 | 实际每rank prefill块 | SWA比例 | 负载 |
| --- | --- | --- | --- | ---: | ---: | --- |
| dpaeager | TP1/DP2 | EP2/TP1 | eager | 512 | 0.1 | 历史144/792输入、512输出 |
| dpagraph | TP1/DP2 | EP2/TP1 | graph | 512 | 0.1 | 同上 |
| epgraph | TP2/DP1 | EP2/TP1 | graph | 512 | 0.1 | 同上，当日对照 |
| epprefill | TP2/DP1 | EP2/TP1 | graph | 8192 | 0.5 | 约1024/4096/7168输入、64输出 |
| dpaprefill | TP1/DP2 | EP2/TP1 | graph | 8192 | 0.5 | 同上 |

每个提示预热1次、正式3次；共48条性能请求，36条正式、12条预热。DP单请求逐条发送，轮流固定DP0/DP1，逐rank明细保留；三个正式重复的rank样本数不相等，合并中位数仅作配置级探索，不能作为rank等价证明。另采两段各8个forward步骤的CPU/NPU时间线，使用单独32输出请求，不进入正式统计。每模式按DP0/DP1运行算术和JSON格式观察，记录答案但不以JSON包装不符中止性能。空输出、不可用内容、接口/通信/设备异常、缺失token或计时证据仍失败即停。

所有阶段：每层全局常驻64专家，两卡各32个完整专家，CPU仅rank0计算64–255；96线程/4池、seed20261111、单请求、max-total-tokens16384、context8192、mem-fraction0.81。DP全局max-running-requests2，对照1，均每rank一个请求槽。关闭streaming prefill/dynamic resident/side stream/radix cache。DP会将CLI分块除以2，长输入CLI16384以得到每rank8192。长输入SWA池预期8192 token，最后一个输入不超过7168、输出64，留出512窗口和128页等余量。运行前检查池几何，OOM或启动失败记录为容量结果，不自动改参数重试。

## 指标与比较边界

性能使用原生 `/generate` token stream；先用 `/tokenize` 核对历史token IDs，再冻结长输入token IDs，所有配置复用。保留消息、输入IDs、采样参数、每次token计数及到达时间、最终正文和SHA-256、SSE结束与usage。新轮按token计数而非首个非空正文块测TTFT，故历史OpenAI正文块TTFT仅作背景，主比较来自本轮同接口测量。

- TTFT：请求开始到首个输出token事件；首个事件必须是第1个token。
- 平均TPOT：第1到第N个token事件间隔除以N−1；decode token/s为倒数。保留合并事件标记，不能把SSE包个数当token数。
- 服务端prefill：启用metrics，读取原有 `forward_entry_time` 和 `prefill_finished_time` 的差值，并计算实际输入token/s。这是scheduler观察的prefill forward墙钟区间，含主机/排程与完成开销，不是纯NPU核时间，也不同于TTFT。
- 每阶段保存server_info、HBM、系统内存、NUMA统计、进程RSS。扩大prefill块和SWA比例的阶段单列，不能把它们与512块混成只改变并行配置的对照。
- 每提示仅3个正式样本，给中位数和范围；不声称具有代表性的P95。每配置单次加载、顺序运行，均为探索性结果。

Profiler在正式请求后开启8步，按固定源码中CPU/GPU活动映射采集CPU和NPU，单独保存。原生KT线程内部的工作不一定在PyTorch trace中完整展开；不把空白时间直接认定为某个硬件瓶颈。Profiler不可用但服务健康时记录缺口，不以profiler请求耗时代替性能；运行异常仍停止。原始trace留远端profiles，自动回收测量与日志及trace文件清单。

## 后台与停止

预计约45–120分钟，远端上限3小时；隐藏本机守护沿用既有Taishan-NpuCooling只读状态和≥90°C持续10秒停止策略，SSH恢复75秒、远端心跳失联90秒保持。原服务初始停止；任务结束或失败后停止所有实验容器、核对PID0、端口释放、无NPU计算进程并保持原服务停止。原始失败记录保留；不自动恢复为常驻服务，不发布或commit。

启动短查后结束本轮，不持续等待。用户叫回再读取原始结果、复算并判断后续CPU/NUMA、热专家放置或更大模型的容量实验。此后台任务不会自行下载更大模型、扩大并发或更换量化格式。
