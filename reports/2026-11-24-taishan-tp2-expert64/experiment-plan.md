# TaiShan：TP2 将常驻专家从 32 增至 64

日期：2026-11-24（北京时间）。用户已要求按所附讨论执行部署实验，授权本轮隔离目录、临时容器及其测试。长任务后台运行，启动核查后结束本轮；完成后由用户再次调用，再复核结果。后台流程结束不代表实验已人工验收。

本轮检验 SGLang + kt-kernel 的 TP=2/EP=1、K=64：每层前 64 个专家的中间维度由两张 NPU 分片，CPU 计算其余 192 个专家。K 是两卡共有的逻辑专家集合大小。CPU 的完整 MXFP4 压缩存储可能仍约 137 GiB，不能假定 RAM 随 K 等比例下降。

使用历史实机环境镜像 `sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e`，SGLang `ff46d660d16cbf11ef393c80cc6b3ab7f6259780`（Pan-Boyi fork），匹配的 KT `c9873312643b615a717832946fba0346e69ab035` 归档和既有 W8A8/MXFP4 权重。源码单独归档后施加原 TP2 opt-in 补丁和仅启动期执行的 mask/张量维度断言，不更改请求前向计算。记录基础、补丁及实验文件的 SHA-256，运行时核对 sglang 和 kt_kernel 导入路径。使用现有文件，不下载或替换权重。

## 顺序与固定条件

1. 空闲资源、模型文件、镜像及散热检查；新容器先核查正常导入后的 KT 扩展和 LLAMAFILE API，再执行两 rank HCCL 求和。环境导入检查也后台执行，独立阶段上限 300 秒，失败保留具体环境日志。
2. TP2/K64 eager：加载、43 层两个 rank 的常驻 mask/权重专家维度审计、CPU 仅 rank 0、三条自然停止冒烟请求。
3. TP1/K32 decode graph 对照。
4. TP2/K32 decode graph 对照。
5. TP2/K64 decode graph 目标配置。

三个图模式均执行同一份五题短功能检查，之后使用 11 月 11 日冻结的两条请求（144/792 输入 token），每条预热一次、正式测量三次，每次强制生成 512 token。共 6 条预热、18 条正式性能请求。tokenizer 结果必须等于冻结 token IDs。性能统计包括 TTFT、客户端输出 token 间隔速度与总耗时，同时保留完整 SSE、usage、finish reason 和文本哈希。单配置仅一次独立启动，因此结果是探索性对照，不能排除运行次序及冷启动间的变化。

固定：96 CPU 线程、4 线程池、单请求、seed=20261111、关闭 streaming prefill/dynamic resident/side stream/radix cache，mem-fraction=0.81、context-length=8192、chunked-prefill=512、max-total-tokens=16384。测试输入加输出最多 1304 token，低于既有短测范围；不把配置上下文上限当作长上下文验收。

启动日志必须显示全部 43 层、每个 rank 的 resident 数、CPU 专家数、CPU owner 与实际 w13/w2 张量形状；TP2 每 rank intermediate=1024、full=2048。图模式每个 rank 必须成功捕获 decode 图。每阶段采集 npu-smi、free/numastat、容器进程 RSS 和模型加载显存日志；多进程 RSS 的总和含共享页，不能直接视为独占 RAM。/proc/meminfo 的 MemAvailable 差值也受页缓存等影响。

## 失败、恢复与验收边界

部署前原 `dsv4-kt-20261104` 已停止、没有运行中的容器；流程记录原状态并在结束后保持停止。如果启动前原服务已被其他操作启动，控制器直接拒绝执行，避免改变它。所有新容器设置 restart=no，仅监听 127.0.0.1:18082；全部阶段结束或软件失败都停止实验容器并核实退出、端口释放及 NPU 无进程。保留容器、源码和日志供检查。遇失败即停，不自动降低 K、改变量化或改走官方 main。

11 月 24 日用户纠正后，以 taishan 项目现行策略替换本计划最初复用的 70°C 立即停止规则：同一张物理 NPU **≥90°C 连续 10 秒**才触发温度停止，低于 90°C 即重置该卡计时，两个卡分别计时；采样中断超过 15 秒不计作持续高温。70°C 与 89°C 的健康读数不停止实验。设备缺失或健康异常、遥测连续失败及心跳过期的停止保护继续保留。

工作站已有 `Taishan-NpuCooling` 任务（`E:/Coder/vibe-coding/taishan/services/npu-cooling/service_daemon.py`）：70°C 全速、67°C 连续 120 秒恢复 auto。它负责全部风扇调节；本轮不启动、停止或修改该服务，不设置 iBMC 风扇，不要求风扇固定为 auto，不保留旧 4500 RPM 的额外停机门槛。本机实验进程只读其 `C:/Users/tobeo/AppData/Local/taishan/npu-cooling/status.json`，要求服务 running、有效样本不超过 30 秒、现行全速策略为 70°C，记录原始状态并更新远端心跳。读取相同缓存样本不增长持续高温计时。远端独立读取 NPU，执行同一个 90°C / 10 秒规则，并校验本机心跳。

策略依据是 `E:/Coder/vibe-coding/taishan/ascendbench/benchmarks/run_qwen38_baseline.py` 的 `temperature_limit_c/temperature_hold_seconds` 逻辑及 `tools/launch_qwen36_graph_baseline.py` 的 `90/10` 配置；有效版本和源码哈希记录在 `policy-alignment.json`。本地和远端执行热规则、SSH 传输恢复和 checkpoint 完整性测试后才启动实验。远端流程上限三小时，本机保护含清理余量。stdout、stderr、status.json、commands.jsonl、thermal.jsonl 和 cooling-service-samples.jsonl 持久保存。

## SSH 传输中断后的续跑修订

`expert64-20261124-cooling90` 已完成 K64 eager、TP1/K32 图模式五题检查及八条性能请求（两条预热、六条正式），之后在 TP2/K32 加载初期因本机连续两次 SSH 心跳命令超时而中止。设备健康、温度与现有温控服务没有触发停止规则。

本机将实际冷却遥测失效与 SSH 传输错误分开计数，心跳更新和状态读取合并为一次 SSH 调用。真实遥测连续失效继续停止；短暂 SSH 超时在自上次心跳确认起最多 75 秒内重试，并显示 `reconnecting`，不再因两次短暂命令超时直接写 abort。远端无心跳 90 秒停止的保护保持。记录单独的 `transport.jsonl`，不把 SSH 错误写成温度告警。

新轮 `expert64-20261124-ssh-resume` 保留旧目录，从旧轮只读核验并复制完整的 K64 eager 和 TP1/K32 两阶段，重新执行环境与 HCCL 探针后继续两个 TP2 图模式阶段。复用前必须核对镜像、forward candidate、SGLang commit、native 扩展、权重 metadata、提示 token、启动命令及工作负载 SHA，逐请求复算输出哈希、512 token、SSE/时间戳/速度，并要求旧轮清理通过。只复制已完成阶段，旧 abort/alarm/status 不带入新轮；记录 `checkpoint-reuse.json` 的来源文件哈希，并在汇总中显式标注复用。旧日志不改写，复制后的日志不再被不存在的同名新容器覆盖。

完整比较仍为 18 条正式请求，其中 TP1 的六条来自此前完整阶段，两个 TP2 各新执行六条。跨 SSH 中断的时间间隔与顺序效应必须作为性能解释边界，不能称为一次连续或交错运行。

本轮验收范围是部署、专家驻留、短任务、SSE、图执行及同负载探索性性能。TP1/TP2 原有固定历史数值差异仍待解释；K32→K64 还改变了部分专家的计算后端与精度。短题通过或速度提升均不证明数值等价、独立任务质量、长上下文或生产稳定性。不自动发布评论、提交代码或上线常驻服务。
