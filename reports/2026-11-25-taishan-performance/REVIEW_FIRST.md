# 首轮：正式DP测量完成，profiler自动停止触发NPU同步超时

运行 `performance-20261125-093427-first`。原状态保持failed、cleanup_verified=true。终态的Connection refused发生在scheduler已经退出之后，不是首次错误，也不是SSH连接失败。以下时间按容器日志本身顺序，不与本机/宿主机有偏移的时钟直接混排。

## 直接证据

1. environment、默认设备/请求池、两rank HCCL和实际DP合并probe通过。
2. dpaeager、dpagraph完成权重加载、逐层专家和注意力拓扑审计及各8条固定512输出请求。16条中12条正式、4条预热，全部在profiling开始前结束，并已有after-resources快照。
3. dpagraph.log的02:20:11记录两rank开始profiling，start_profile返回200，随后独立32输出请求开始。
4. 02:20:14两rank自动Stop profiling，记录Stop profiler while current state is RECORD。02:23:04均在AclrtSynchronizeDeviceWithTimeout报告107020、event wait timeout，超时150秒。
5. 02:23:22 DP1的scheduler在后续forward张量提交时退出；其后服务被进程树清理，客户端流中断，异常处理中的health检查再报Connection refused并覆盖了原错误。

固定源码scheduler.run_batch在启动新forward前调用_profiler_batch_predicate；num_steps=8使该函数在第8步边界调用_stop_profile，而请求尚未完成。故障明确出现在请求内自动停止profiler后的NPU同步阶段。涉及NPU graph、CPU专家回调、DP/HCCL的具体未完成事件依赖尚未定位，不能把它写成已证实的某个算子或HCCL库缺陷。权限警告不是现有证据中首次致命异常。

## 已完成的数据

独立从原始token事件重新计算TPOT、decode与prefill，检查16条矩阵身份、DP路由、固定input IDs、seed、temperature、512输出、length结束及正文SHA-256，均通过。

| 配置 | 输入token | TTFT中位数 | TPOT中位数 | decode中位数 |
| --- | ---: | ---: | ---: | ---: |
| DP-attention eager | 144 | 2.447 s | 369.39 ms | 2.707 tok/s |
| DP-attention eager | 792 | 10.202 s | 364.06 ms | 2.747 tok/s |
| DP-attention graph | 144 | 2.142 s | 87.95 ms | 11.369 tok/s |
| DP-attention graph | 792 | 9.929 s | 87.31 ms | 11.453 tok/s |

这些仅是本轮原生token接口下的先完成阶段，尚无当日TP2对照或长输入结果，不能将整个实验标为完成。JSON格式观察仍失败、算术通过；按用户指定规则分别记录，没有用格式失败阻止性能。每提示三次、DP rank分布不等、单次加载、顺序运行，结论限探索性性能。核验明细在原运行目录reviewed-partial.json。

## 修复与继续

正式性能控制器移除profiling API，输出deferred及明确原因；不通过提高NPU超时或关闭watchdog来掩盖同步问题。后续trace采集须在独立进程/实验中验证退出边界。异常记录增加完整traceback和失败阶段，避免只留下继发的HTTP错误。

新增resume_performance.py：逐项核对launch、运行代码、候选源码、镜像、原生扩展、权重元数据/大小、输入和请求参数后，才逐字节复制先完成的16条请求及资源/日志。源日志原样保留，dpagraph的验收明确限定为首次Profiling starts之前的前缀，不将后续失败日志伪装成完整成功。检查来源bundle启动后没有变化，复制文件记录SHA。新轮重跑环境与HCCL，仅执行epgraph、epprefill、dpaprefill；比较属于跨运行顺序比较。

协议、拓扑、温控及checkpoint测试共24项（暂存/远端缺少本机源证据时1项明确跳过）；另在实际本机16条记录上执行复制与参数改变拒绝回归通过。旧状态、日志、trace不修改。

## 清理与trace权限

全部首轮容器已退出、PID0、OOMKilled=false，18082释放，两卡无计算进程、健康OK，原服务保持停止。没有高温保护触发证据，也没有模型OOM证据。

普通用户生成的profile-inventory为空，不能据此断言trace不存在。sudo只读清单确认264个文件、284,621,738字节，包含两rank的trace_view.json和kernel_details.csv；目录由容器root创建，普通用户不能遍历完整内容。补充清单保存在原本机运行目录profile-inventory-root-read.json，原清单保留。因采集停止超时且日志有数据丢失警告，这批文件只作异常诊断，不用于正式性能归因。
