# TaiShan 双卡性能与 prefill 扩展

追加更正：DP0/792输入的16.85/17.01 tok/s两条记录虽有512个token计数，正文均只有 `This`。它们是输出异常下的计数吞吐，不能作为正常回答速度；详见[结果报告中的更正](REVIEW.md)。原始数据及三次中位数保留，不静默剔除。

2026-11-25 继续轮 `performance-20261125-113036-profiler-resume` 已完成并独立回读，见[结果报告](REVIEW.md)。48条请求中36条正式、12条预热完整；TP2/EP2短负载decode约15.5 tok/s，DP-attention graph中位数约11.4 tok/s。两种拓扑的1k/4k/7k prefill均约83–87 tok/s，7k首token等待约83秒。DP0的792输入另有17 tok/s记录，分rank输出和速度差异保留，不能只报合并中位数。

首轮profiler自动停止触发107020，见[故障复盘](REVIEW_FIRST.md)。继续轮校验后复用先完成的16条请求，新跑其余32条；正式性能流程关闭profiler，trace未验收。81项远端证据和4个部署源码哈希核对通过，容器与端口已清理，原服务保持停止。完整参数见[experiment-plan.md](experiment-plan.md)。

本轮目标是TTFT、TPOT、prefill/decode速度与输入容量，不以固定token数值定位作为前置。保留原JSON题的输出观察，格式失败不阻断性能；运行/通信异常或缺失测量数据仍失败即停。

顺序为：

1. 固定环境、默认设备与请求池探针，双rank HCCL和实际DP合并probe。
2. attention TP1/DP2 + MoE EP2，eager与graph各一次加载；历史144/792输入、固定512输出。
3. attention TP2 + MoE EP2，graph同日对照、相同负载。
4. 两种attention拓扑分别加载宽prefill配置，约1024/4096/7168输入、固定64输出。

汇总仍为48条性能请求，其中36条正式、12条预热；继续轮复用12条正式/4条预热，新增24条正式/8条预热。短trace因首轮停止时107020超时暂缓。配置均为每层两卡各32个完整专家、CPU仅rank0、96线程/4池。DP请求逐条发送，不开展并发压力测试。原运行与继续轮时间不同，比较保留跨运行顺序测量限制。

原生 `/generate` token流记录token计数到达时刻，计算TTFT及平均TPOT。服务端prefill使用原有scheduler起止时间，单列forward墙钟耗时及token/s，不把TTFT倒算为纯prefill。新接口的首token口径与历史OpenAI首个非空正文块不同，历史数据只作背景；本轮同接口配置互相比较。长输入阶段单独扩大实际prefill块到8192、SWA比例到0.5，不能与短配置混作单因素对照。

## 历史运行与证据

历史启动器PID为13724、实际本机守护PID为15060、远端控制器PID为3112477，后台流程已经结束。运行名称含一次性解析的北京时间秒级时间戳，本地、远端、容器名一致。

首轮已完成全部环境/通信探针和前两个性能阶段；因profiler故障终止并通过清理，原单卡服务保持停止。继续轮重新执行环境和HCCL，再验证checkpoint身份，确认一致后才跳过已完成阶段。

- [继续轮实时状态](runs/performance-20261125-113036-profiler-resume/status.json)
- [继续轮远端启动信息](runs/performance-20261125-113036-profiler-resume/launch.json)
- [本机标准输出](guard-profiler-resume.stdout.log)、[标准错误](guard-profiler-resume.stderr.log)
- [首轮原失败状态](runs/performance-20261125-093427-first/status.json)
- 本机运行目录：`reports/2026-11-25-taishan-performance/runs/performance-20261125-113036-profiler-resume/`。
- 远端目录：`/data/deploy/dsv4-tp2-performance-20261125-113036-profiler-resume`。

```powershell
ssh taishan 'cat /data/deploy/dsv4-tp2-performance-20261125-113036-profiler-resume/status.json'
ssh taishan 'tail -n 30 /data/deploy/dsv4-tp2-performance-20261125-113036-profiler-resume/controller.stdout.log'
ssh taishan 'sudo -n docker logs --tail 40 dsv4-tp2-performance-20261125-113036-profiler-resume-epgraph'
```

继续轮模型容器后缀为epgraph、epprefill、dpaprefill，均已停止；前两个阶段从首轮复用。上述Docker命令读取留存日志，不会启动模型。

原始请求在requests.jsonl，内容观察在functional-observations.jsonl，输入冻结在frozen-workloads.json和对应SHA文件；server-info与resources保存配置、HBM、RAM/NUMA。checkpoint-reuse.json记录复制证据SHA与原日志验收边界；源日志原样复制，包含之后的profiler失败，不把它隐藏成完整成功。汇总summary.json，失败failure.json，退出验收cleanup.json。

首轮原始trace保留远端profiles，sudo只读清单确认264文件、约285MB；普通用户权限不足导致原profile-inventory误为空，补充清单在首轮本机目录profile-inventory-root-read.json。因采集退出超时和数据丢失警告，这批trace不用于正式性能归因。

后台保护已结束，本轮不再需要持续实验心跳。未改动既有Taishan-NpuCooling；继续轮最高72°C/66°C，未触发≥90°C持续10秒停止。服务保持初始停服状态，本次已完成数据和探索性性能回读，未自动启动后续实验。

[独立复核程序](verify_results.py)、[分rank指标及比较](runs/performance-20261125-113036-profiler-resume/reviewed-results.json)、[远端证据哈希](runs/performance-20261125-113036-profiler-resume/remote-hash-check.json)。

启动前两张NPU健康OK、空闲，RAM可用约495GiB。20项协议、token计时、拓扑审计和温控测试通过；实际固定源码四文件补丁应用及Python编译通过，远端bash语法通过。没有下载新权重、升级环境或启动其他模型。
