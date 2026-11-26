# taishan prefill优化实验

2026-11-25，北京时间。实验已完成并独立回读，见[结果报告](REVIEW.md)：单卡TP1/EP1/K32的4K TTFT从52.85秒降到19.75秒，7K从91.85秒降到20.11秒；1K退化，长输入后的decode也有代价。36条正式+12条预热、真实streaming和清理均通过；完整模型质量未验收。

本轮编号：`prefill-20261125-224217-first`，22:42:17后台启动，23:19完成，23:30独立回读通过。本地证据目录为`runs/prefill-20261125-224217-first/`，远端为`/data/deploy/dsv4-tp2-prefill-20261125-224217-first/`。容器已停止，原服务保持停止，没有启动后续实验。

[实验计划](experiment-plan.md)记录矩阵、固定项、异常处理和温控规则。[公开参考](PUBLIC_BENCHMARKS.md)记录A3数据及可比性限制。上轮双卡结果保留在[原结果报告](../2026-11-25-taishan-performance/REVIEW.md)。

## 运行与数据

启动器为`run_guarded.py --run-id prefill-YYYYMMDD-HHMMSS-first`。后台启动元数据写入`background-launch.json`；一次解析北京时间运行编号，本地`runs/<run-id>/`、远端`/data/deploy/dsv4-tp2-<run-id>/`和容器名共用该编号。远端路径沿用既有控制器前缀，实际配置为TP1。

- 本地`runs/<run-id>/status.json`：观察器状态；`remote-status.json`：远端最近状态。
- 远端`status.json`、`controller.stdout.log`、`controller.stderr.log`：执行状态与控制器日志。
- 远端`logs/baseline.log`、`logs/stream.log`、`logs/bundle.log`：模型加载、图捕获和streaming执行证据。
- `requests.jsonl`：逐条输入/原始输出IDs、正文、TTFT/TPOT、prefill和流式审计；`summary.json`：正式请求汇总。
- `kernel-probe.json`：AscendC构建与加载；`*-resources.json`：资源快照；`cleanup.json`：清理验收。
- 本地`reviewed-results.json`：逐次指标独立复算、证据完整性与资源/温度核验；`remote-hash-check.json`：远端SHA和实时停止状态快照。用[核验程序](verify_results.py)可离线复算。
- 终态后后台观察器自动将证据下载到本地`runs/<run-id>/remote/`；原始部署源码归档留在远端。

后台流程自行顺序执行和失败清理；启动后只检查正常起步，不持续等待完成。完成后由用户再次要求回读。
