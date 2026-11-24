# 注意力TP1 + MoE EP2/TP1

2026-11-24 定点诊断 `attn-tp1-ep2-20261124-214932-format-diagnostic` 已完成回读，见 [诊断结果](REVIEW_DIAGNOSTIC.md)。原JSON排序题在eager/graph、DP0/DP1各重复两次，8次均返回相同非法JSON；纯数字排序和算术控制题16次全部正确。部署、通信及图重放通过，模型质量仍未验收；没有512 token性能数据，原服务保持停止。

首轮已完成权重和两类86条审计，但全局请求上限1经DP2划分为0导致启动失败，见 [失败分析](REVIEW_FIRST.md)。当前全局上限为2、每rank为1，客户端仍逐条请求；环境阶段先执行实际请求池公式回归探针。

第二轮请求池已修复，但首次warmup误入依赖PyNCCL的reduce_scatterv，见 [第二轮分析](REVIEW_REQUEST_POOL_FIX.md)。本轮为NPU实验修正统一合并策略，使用一次HCCL AllReduce与DP scatter；新增双卡BF16通信/分发probe，覆盖空闲rank、不等长token与NPU graph重放，通过后才加载完整模型。

| 模块 | 配置 |
| --- | --- |
| 注意力 | TP1、DP2，两卡各有完整副本，单请求在其所属attention rank计算 |
| NPU2 / MoE rank0 | 完整专家0–31 |
| NPU5 / MoE rank1 | 完整专家32–63 |
| MoE内部TP | 1 |
| CPU | 仅rank0执行64–255，RAM仍保存完整MXFP4权重集合 |

CLI为 `--tp 2 --dp 2 --enable-dp-attention --ep 2 --moe-a2a-backend none`。此处TP2是两进程组；实际attention TP为1，由逐层运行时审计确认。MoE前gather token，两个NPU和CPU分别计算各自专家，求和后scatter回attention所属rank。每卡注意力不再切半，HBM可能增加。

当前诊断重跑环境/默认设备/请求池探针、HCCL及实际合并probe、eager和graph模型；每模式执行原JSON排序提示、额外纯数字排序/算术控制题，每题固定路由DP0/1并各重复两次，共24条短请求、每条max_tokens64。原提示不改、不提取错误包装中的数组、不开始性能测试。既有两类86条审计保持，decode日志间隔1用于核对实际重放。

保持96 CPU线程/4池及原权重。DP attention将chunked-prefill-size除以2，故传1024以保持实际每rank512。温控沿用工作站服务，实验>=90°C持续10秒停止；SSH/心跳保护和失败即停保留，完成后原服务保持停止。

后台任务已结束；历史启动器PID为31204（实际守护进程及远端PID见运行状态和launch.json）。[运行状态](runs/attn-tp1-ep2-20261124-214932-format-diagnostic/status.json)、本目录 `guard-format-diagnostic.stdout.log` / `guard-format-diagnostic.stderr.log`、`background-launch-format-diagnostic.json`。完整运行采样保存在 `runs/`，模型源码快照和日志不纳入Git。

```powershell
ssh taishan 'cat /data/deploy/dsv4-tp2-attn-tp1-ep2-20261124-214932-format-diagnostic/status.json'
ssh taishan 'tail -n 30 /data/deploy/dsv4-tp2-attn-tp1-ep2-20261124-214932-format-diagnostic/controller.stdout.log'
```

诊断结果已独立回读，原质量门槛及 correctness_accepted=false 保留。后续建议固定token历史对比attention TP1/TP2的logits以定位首次分歧；本轮未自动启动新实验、提交或发布。
