# 速度对照首轮：单 rank 日志格式被检查器误判

核查日期：2026-11-11（北京时间）。运行目录为本机 `runs/speed-20261111-first/`、远端 `/data/deploy/dsv4-tp2-speed-20261111-first`。本轮尚未生成正式速度数据。

TP1 的首个实验容器实际完成了加载和 decode 图捕获：日志在北京时间 15:00:07 记录 `Capture target decode NPU graph end. elapsed=4.48 s`，15:00:11 记录服务已就绪。但 SGLang 在 TP=1 时不附加 `TP0` 日志标签，控制器却只接受 `TP0] Capture target decode NPU graph end`，从而误报 `No decode graph for tp1a rank 0`，在任何质量门槛或正式测速前停止实验。

这是实验检查器的格式假设错误，没有证据说明单卡图捕获失败，也没有 TP2 速度结果。控制器进入 `restoring_original`，`benchmark_error` 字段保留这条实验错误；外层本机 `phase=running` 表示保护与恢复流程仍在执行，不能据此把基准当作还在测速或已恢复。

## 修正与验证

在 `workload.check_graph()` 中，先核对解析后的 `tp_size`，TP=1 接受不带 rank 的捕获结束日志，TP=2 仍要求两个 rank 都有成功标记。使用本次真实 TP1 日志与此前成功的 TP2 日志复核，两者均通过；删除捕获标记、缺失 TP1 rank 标记或错误 TP 大小时均正确拒绝。Python 语法检查及 `git diff --check` 通过。未修改模型、权重、采样、缓存或性能矩阵。

## 有序重跑

本机隐藏后台控制器 `retry_after_restore.py` 已启动（PID `24628`），状态见 `runs/retry-graph-check-controller/status.json`。它最多等待原控制器完成：只有旧任务以预期的日志误判失败、同时记录 `original_restored=true`，且原 API `/health=200` 时才启动修正版。其他失败或恢复未成功则停止，不强行打断原服务的恢复过程。

重跑独立保存为 `runs/speed-20261111-graph-check/`，远端 `/data/deploy/dsv4-tp2-speed-20261111-graph-check`。新测速控制器的 stdout/stderr 在 `runs/retry-graph-check-controller/benchmark.stdout.log` 与 `benchmark.stderr.log`。继续使用原事前协议的 ABBA、144/792 输入、512 输出和热保护，完成后恢复原服务；本次失败记录保留。是否已开始新测速及最终结果以对应状态文件为准，不能把排队启动当作测速完成。

后续启动核查：旧控制器最终记录 `failed`、`original_restored=true`，队列健康门槛通过后已启动新任务；新本机 PID `16924`、远端 PID `1151373`。短暂检查显示处于停止原服务阶段，两卡 `OK`、54°C/53°C，标准错误为空。新任务仍未完成测速验收。
