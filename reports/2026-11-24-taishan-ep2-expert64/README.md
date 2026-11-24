# EP2：NPU2/NPU5 各放不同的32个完整专家

2026-11-24 最终运行 `ep2-expert64-20261124-graph-resume` 已完成并回读复核，见 [结果报告](REVIEW.md)。eager三题、图模式五题和6条正式请求通过；两卡各32个不同完整专家的驻留与图重放通过。相对此前TP2/K64，decode低约1.2%–2.0%，总耗时长0.9%–1.9%，HBM基本一致，未观察到性能收益。完整数值及任务质量仍未验收，原服务保持停止。前两轮问题保留于 [首轮记录](REVIEW_FIRST.md) 和 [第二轮记录](REVIEW_DEVICE_FIX.md)。

| 执行位置 | 每层路由专家 | 权重 |
| --- | --- | --- |
| 物理 NPU2 / rank0 | 0–31 | 每个专家完整中间维度2048 |
| 物理 NPU5 / rank1 | 32–63 | 每个专家完整中间维度2048 |
| CPU（仅rank0执行器） | 64–255 | 原MXFP4，存储仍保留完整集合 |

注意力TP2、MoE EP2/TP1。CLI仍以全局常驻总数 `--kt-num-gpu-experts 64` 表达，隔离wrapper派生每卡32；通过 `KT_NPU_EXPERIMENTAL_EP2=1` 明确开启。token输入复制到两rank后各自筛选所属专家，最终AllReduce加和；不使用AllToAll。CPU跳过两卡专家并集，CPU结果仅rank0加入一次。

仅修改隔离 SGLang 副本中的 wrapper和专家ID映射，原服务、镜像、权重及温控服务不改。每层/rank必须记录实际常驻区间、完整张量形状、唯一CPU所有者，并核查checkpoint确实访问各自分配的32个全局专家ID。EP2直接参数放行不足以保证这些条件，故本轮属于实验性适配。

顺序：已有CPU布局测试和温度测试、环境导入、HCCL、eager三题、图模式双rank捕获与五题，最后两种提示各预热一次/正式三次、每次512输出。最多6条正式性能请求；失败即停。CPU布局数学检查不代表真实专家算子数值已经验收，完整logits/任务质量仍需后续复核。

本地17项布局/源码、图捕获解析及checkpoint检查通过，包括真实固定源码、真实图模式日志及旧轮eager证据；远端运行这些检查（无本地源码快照时两项明确跳过）和既有13项温度测试。真实镜像环境步骤另执行 `device_probe.py`，提取生成构造及CPU映射门禁，在非CPU默认设备下先复现旧失败，再验证两个rank的CPU表归属。launch.sh 首轮的远端 `bash -n` 已通过。运行只读记录工作站现有 Taishan-NpuCooling 状态，>=90°C持续10秒停止，远端90秒无心跳停止及SSH重试保持。后台结束后保持原单卡服务停止，并清理实验容器及端口。

状态：[runs/ep2-expert64-20261124-graph-resume/status.json](runs/ep2-expert64-20261124-graph-resume/status.json) 已为complete。历史启动器PID见本地 `background-launch-graph-resume.json`，远端PID见运行目录 `launch.json`。本机stdout/stderr为本目录 `guard-graph-resume.stdout.log` / `guard-graph-resume.stderr.log`；完整采样保存在运行目录。实验心跳已结束，原服务保持停止。

```powershell
ssh taishan 'cat /data/deploy/dsv4-tp2-ep2-expert64-20261124-graph-resume/status.json'
ssh taishan 'tail -n 30 /data/deploy/dsv4-tp2-ep2-expert64-20261124-graph-resume/controller.stdout.log'
ssh taishan 'sudo -n docker logs --tail 40 dsv4-tp2-ep2-expert64-20261124-graph-resume-ep2graph'
```

代码与结果报告纳入Git；`runs/`、完整日志、源码快照及后台启动元数据保留本地。本轮未测eager与graph的同负载速度对照，详见结果报告中的性能边界。
