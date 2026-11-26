# 教程性能表参数复测

**已完成并回读通过**，见[最终结果报告](REVIEW.md)。硬件适配长档于15:05结束，16K/32K各3正式+1预热；与前一批短档合计18正式/6预热、18000逐token事件复核通过，149项远端文件SHA及4项源码SHA一致。16K/32K首正文约21.35/24.26秒，decode9.34/8.49 token/s；32K请求后HBM约59.98GiB。两卡及容器清理通过，没有启动下一轮。下文保留各阶段历史状态，以本段和最终报告为准。

最新现场核验（远端北京时间14:57:23）：`tutorial-20261126-144145-capacity-parser-fix` **正在正常测量**，没有failure.json或运行异常/回退。16K已完成1预热+3正式、各1000输出；3条正式的3000逐token事件、原始IDs、正文SHA和43层streaming复核通过，无正文异常标记。32K已完成43层streaming prefill并进入graph decode，正在首次预热，完整矩阵尚未完成。现场证据保存在本轮本地 `live-verification.json`。早期 `110255-first`、`111251-swa-audit-fix`、`143038-hardware-long` 均是历史失败目录，状态应保留，不能用其判断最新任务。当前观察器PID30188，控制器PID2532285；本次核验未重启或干扰任务。

14:41:45（北京时间）修复容量日志解析后，以相同硬件参数重新后台启动 `tutorial-20261126-144145-capacity-parser-fix`，本地隐藏观察进程PID30188。上一轮实际full73728/SWA36864已初始化成功，因检查脚本将预算和上限重算两条日志误认歧义而在请求前停止，详见[解析误报复盘](REVIEW_CAPACITY_PARSER.md)。本轮只改核验，不修改模型计算或参数；状态和日志以当前background-launch.json和对应runs目录为准。

2026-11-26 14:30:38（北京时间）按用户要求后台启动硬件适配长输入实验 `tutorial-20261126-143038-hardware-long`，本地隐藏观察进程PID6820。参数见[硬件适配计划](hardware-plan.md)，清单见[hardware-profile.json](hardware-profile.json)。只补测16K/32K，K28、mf0.775、SWA0.5、总token上限73728；实分配池须在请求前通过容量门槛，后台启动不代表通过。进度看对应runs目录的status.json，日志路径记录在background-launch.json。原table12结果独立保留，不混作同参数矩阵。

最新回读：第二轮已失败并清理，见[16K准入超时诊断](REVIEW_SWA_TIMEOUT.md)。table12的4档完成（12正式+4预热）；16K的15568输入所需SWA15696位置超过实际池14208，被调度容量门槛拒绝并等到HTTP600秒超时；32K未启动。部分计时已复算归档，本次没有重跑。

2026-11-26。按用户要求使用物理NPU2。配置与验收口径见[实验计划](experiment-plan.md)，公开文档和测试脚本固定版本见[sources.json](sources.json)。当前只有准备/启动记录，尚无验收结果。

11:02:55（北京时间）已后台启动 `tutorial-20261126-110255-first`。本地隐藏观察进程PID22492，远端控制器PID1993709；初步检查已通过远端测试、权重/设备预检及隔离源码准备，正在environment probe。远端目录 `/data/deploy/dsv4-tp2-tutorial-20261126-110255-first`。NPU5已有vLLM容器保持运行。两机时钟观察有约25秒差异，保留各自时间，不用跨主机时间戳相减计算性能。

首轮随后因审计脚本误认SWA默认值失败，未发送性能请求，已清理并下载证据，见[失败复盘](REVIEW_FIRST.md)。修正模型默认值检查和原始文本分词接口后，以新的 `swa-audit-fix` 编号重新后台运行；最新编号、PID与日志路径以 `background-launch.json` 为准。重跑前两卡无计算进程、无运行容器；负载仍逐请求记录。

重跑于11:12:51启动：`tutorial-20261126-111251-swa-audit-fix`，本地隐藏观察进程PID3336、远端控制器PID2027835。21项远端检查通过，控制器已起步，尚无性能验收结果。随后观察到其他任务创建 `mixedsched-20261126-1026-short-first-a` 容器；本轮保留同机负载快照，不将首次空闲观察推广为全程空闲。

使用 `python run_guarded.py --run-id tutorial-YYYYMMDD-HHMMSS-first` 后台启动。一次生成北京时间run ID，本地和远端共用；远端目录沿用既有控制器前缀 `/data/deploy/dsv4-tp2-<run-id>`，实际为TP1。

- `background-launch.json`：隐藏后台观察进程PID、运行编号及日志路径。
- `runs/<run-id>/status.json`、`remote-status.json`：观察器及最近远端状态。
- 远端 `status.json`、`controller.stdout.log`、`controller.stderr.log`：控制器状态及错误。
- 远端 `logs/table12.log`、`table19.log`、`table39.log`：三组参数的加载和执行日志。
- 远端 `requests.jsonl`、`warmup-*.json`、`frozen-cases.json`：原始请求、输出IDs、公开口径及逐token计时。
- `summary.json` 为自动汇总，仍需人工回读；OOM、回退和清理失败记入 `failure.json`。

后台终态后既有观察器自动下载证据到本地 `runs/<run-id>/remote/`。不持续等待，用户下次要求后再复核与比较。NPU5运行其他服务，同机负载会保留在资源快照中，本轮不能单独隔离硬件代际影响。
