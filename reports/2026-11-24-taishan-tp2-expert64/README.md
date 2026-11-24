# TP2 / K64 部署实验

2026-11-24 最终运行 **`expert64-20261124-ssh-resume` 已完成并回读复核**，见 [结果报告](REVIEW.md)。18 条正式请求、6 条预热、K64 eager 与三个图配置的短功能检查完整；K64 实际驻留及图执行通过。相对 TP2/K32，K64 的 TTFT 缩短 7.5%–9.3%，decode 慢 1.2%–1.5%，每卡多占约 16.16 GiB HBM。完整数值与独立任务质量尚未验收，原始 `correctness_accepted=false` 保留。

TP1 基线和 K64 eager 从前驱完整阶段核验复用，两个 TP2 图模式在新轮执行。这是跨中断的顺序探索性比较；前驱失败记录见 [REVIEW_COOLING90.md](REVIEW_COOLING90.md)，范围见 [experiment-plan.md](experiment-plan.md)。

停止规则保持 **同一张卡 ≥90°C 连续 10 秒**；远端 90 秒无心跳停止保持。风扇完全由现有 `Taishan-NpuCooling` 服务管理，本轮只读它的状态。短暂 SSH 失败显示 `reconnecting`，在自上次确认心跳起 75 秒预算内重试，单次心跳更新与状态读取合并。

首轮的导入检查错误见 [REVIEW_FIRST.md](REVIEW_FIRST.md)。第二轮 `expert64-20261124-import-fix` 已通过 HCCL、TP2/K64 eager 的 86 条驻留核查与三条短请求，但在 TP1 基线因沿用旧 70°C 停机门槛被中止；已回收复核并保留全部数据，见 [REVIEW_IMPORT_FIX.md](REVIEW_IMPORT_FIX.md)。K64 eager 每卡 HBM 约 42.5 GiB；正式性能对照未完成，不据此声称加速。

使用固定 Pan-Boyi SGLang fork、原 TP2 opt-in 补丁、匹配的 Ascend kt-kernel 和现有 W8A8/MXFP4 文件；启动期额外核查实际专家 mask 和张量维度。目标 `TP=2 / EP=1 / K=64`，两卡合计每层 64 个常驻专家，CPU 计算另外 192 个。后台依次执行 KT 环境检查、HCCL、TP2/K64 eager、TP1/K32 graph、TP2/K32 graph、TP2/K64 graph；图模式共 18 条正式性能请求。

## 已结束运行与证据

- 进程身份：启动器 PID 见 `background-launch-ssh-resume.json`，实际监视进程 PID 见本机状态，远端 PID 见运行目录下 `launch.json`。控制器脱离 SSH 会话运行。
- 实验进程已经结束，模型容器全部停止，原服务保持停服。既有温控服务继续独立运行，本轮没有修改它。
- 远端目录：`/data/deploy/dsv4-tp2-expert64-20261124-ssh-resume`。
- 本机状态：[runs/expert64-20261124-ssh-resume/status.json](runs/expert64-20261124-ssh-resume/status.json)。
- 本机标准输出/错误：`guard-ssh-resume.stdout.log`、`guard-ssh-resume.stderr.log`；样本位于运行目录内的 `thermal.jsonl` 与 `cooling-service-samples.jsonl`，传输错误单列 `transport.jsonl`。

修复后的 `environment_probe.py` 先导入 `kt_kernel`，再取得已注册的 native 扩展，检查公开 factory `__new__` 参数和 LLAMAFILE 后端方法。独立环境阶段允许 300 秒，保留 `logs/environment.log` 和成功后的 `environment-preflight.json`。重试没有安装或升级依赖，没有改变权重、K64 参数或模型计算代码。

远端查询：

```powershell
ssh taishan 'cat /data/deploy/dsv4-tp2-expert64-20261124-ssh-resume/status.json'
ssh taishan 'tail -n 30 /data/deploy/dsv4-tp2-expert64-20261124-ssh-resume/controller.stdout.log'
```

加载期间可以直接看 Docker 日志，依阶段将容器后缀换成 `tp1k32graph`、`tp2k32graph`、`tp2k64graph`：

```powershell
ssh taishan 'sudo -n docker logs --tail 40 dsv4-tp2-expert64-20261124-ssh-resume-tp2k32graph'
```

远端 `logs/<phase>.log` 在阶段健康就绪和停止时抓取；加载中以 Docker 原始日志为准。请求逐次记录在 `requests.jsonl`，启动驻留核查在 `*-startup-audit.json`，资源在 `*-resources.json`，汇总在 `summary.json`，清理验收在 `cleanup.json`。本机守护流程看到终态后会把结果取到本机运行目录的 `remote/` 子目录；这一步是自动收集，不是人工复核。

## 已完成的启动前检查

SSH 按现有配置别名连接，没有读取密钥内容。两卡空闲、64 GiB/卡，RAM 可用约 495 GiB；原容器已停止。固定镜像存在，SGLang 基础 commit 与原 wrapper SHA-256 一致、无已跟踪改动，43 个 GGUF 存在。后台另记录权重分片大小、GGUF magic 与配置文件 SHA-256，复用既有完整权重审计，不重新读取数百 GiB 计算哈希。

本地 Python 语法检查通过；四种 TP/K/执行模式的驻留日志检查器正例和 12 个异常例通过；在实际固定 wrapper 上应用候选补丁并编译通过；冻结请求 SHA-256 与历史一致；`bash -n` 在远端通过。候选 wrapper SHA-256 为 `868864fc5c2a8cdf78d34967e0a14a7e786e8c3a959db1726a89beebd7ec6d28`。

温度、传输和复用检查共 26 项本地测试通过，后台在远端重跑同一组后才启动实验。覆盖温度边界和接线、两次 20 秒 SSH 超时后恢复、75 秒预算、单次心跳/状态查询，以及用实际 TP1 证据验收复用、拒绝不完整请求和变更环境。策略溯源见 [policy-alignment.json](policy-alignment.json)。既有风扇服务保持 PID `7468`，70°C 全速和 67°C/120 秒回差均保持。

复用明细和 SHA 写入新轮 `checkpoint-reuse.json`，新汇总显式标注六条 TP1 正式请求来自 `cooling90`。该比较跨运行中断，保留时间间隔和顺序效应的限制；旧目录与原始状态保持。

本地检查：`python -m unittest discover -s reports/2026-11-24-taishan-tp2-expert64 -p "test_*.py" -v`。其中八项 checkpoint 检查读取归档 `cooling90` 证据；新检出未包含被忽略的运行数据时会明确跳过，其他 18 项可独立执行。`verify_results.py` 同样需要完整运行归档，不能仅凭仓库文档重建原始测量。

本轮已复核端口 18082 释放、两卡无计算进程、原服务仍停止、61 项当前与 21 项来源证据哈希一致。结果及测量口径见 [REVIEW.md](REVIEW.md)，原始 status/summary 不改写为完整正确性验收。
