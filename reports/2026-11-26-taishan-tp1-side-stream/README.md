# TP1侧流与长输出对照

最新：重启轮已完成并回读通过，见[结果报告](REVIEW.md)。24正式+8预热、32000输出IDs/通知完整，140项远端文件及2项部署源码SHA一致。侧流使普通路径decode提高9.5%–10.7%；4K streaming+侧流TTFT19.99秒、decode16.33 token/s、总时长81.18秒，比H0少32.14%。动态驻留关闭，streaming后的初段decode慢、后段恢复；清理通过，没有启动下一轮。

重启记录：北京时间21:08:19隐藏后台启动`tp1-side-20261126-210819-notification-fix`，本地观察PID30396、远端控制器PID3385767，远端22:12:17结束。使用修复后的累计通知校验完整重跑四组，不混用首轮测量；原失败run及其启动元数据归档保留。仅使用NPU2、保留NPU5任务，结果按共享主机条件解释。状态和证据路径见background-launch.json及本轮runs目录；下文为首轮历史记录。

首轮历史：首轮已于远端20:08:56失败并完成清理，原因是h1的4K预热出现一次524→526累计通知跳变，被旧校验器误判为token缺失，见[首轮复盘](REVIEW_FIRST.md)。1000个输出ID和正文完整，H0及H1短档数据保留；S0/S1未运行。本地校验修复通过27项回归和13条真实记录离线回放，当时排查未重启；后续重启见上文。下文首轮启动状态为历史记录，实时状态以当前run的status.json为准。

北京时间19:39:57已隐藏后台启动`tp1-side-20261126-193957-first`，本机观察进程PID21140，远端控制器PID3145825。短查确认23项远端检查、权重/原生扩展身份及AscendC编译加载探针通过，控制器已进入h0模型加载，stderr为空。尚未形成性能或质量验收结果；状态和日志入口见下文，后续阶段由控制器顺序执行。本轮按规则结束前台检查，不持续等待。

2026-11-26，北京时间。用户已要求直接使用物理NPU2启动R1四组实验；NPU5的现有任务保留，同机负载随请求记录。配置、采样、边界和退出条件见[实验计划](experiment-plan.md)，研究依据见[计划复审](../2026-11-26-taishan-research-plan-review/REVIEW.md)。

TP1/EP1/DP1、K32、CPU96线程/4池、graph；144/4096输入、1000输出，h0/h1/s0/s1分别为普通、普通+侧流、streaming、streaming+侧流，动态驻留始终关闭。24正式+8预热，启动或自动汇总不代表结果验收。

后台入口`run_guarded.py --run-id tp1-side-YYYYMMDD-HHMMSS-first`。一次解析北京时间run ID，本地runs目录、远端目录与容器名称共用；远端沿用已有`/data/deploy/dsv4-tp2-<run-id>`前缀，实际是TP1。

- `background-launch.json`：本地隐藏观察进程、运行编号和日志路径。
- `runs/<run-id>/status.json`、`remote-status.json`：本地及最近远端状态。
- 远端`controller.stdout.log`、`controller.stderr.log`、`logs/h0.log`等：控制器、模型加载与执行日志。
- `requests.jsonl`：逐次输出IDs、正文、逐token时间、分窗TPOT及真实streaming审计。
- `*-resources.json`与`commands.jsonl`：共享主机负载和设备采样；`summary.json`为待人工回读汇总。
- `failure.json`、`cleanup.json`：失败证据与只清理本轮容器的退出检查。

后台结束后自动下载证据至对应`runs/<run-id>/remote/`；等待用户再次要求回读，不启动R2或TP2。
