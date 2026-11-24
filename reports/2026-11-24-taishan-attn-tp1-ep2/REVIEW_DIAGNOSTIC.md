# 定点诊断完成：JSON错误跨rank、跨模式稳定复现

2026-11-24 回读 `attn-tp1-ep2-20261124-214932-format-diagnostic`。诊断矩阵24条全部收集，16条回答正确、8条JSON排序均失败。这里complete只表示诊断结束，原性能流程的质量门槛仍失败，`original_quality_gate_accepted=false`、`correctness_accepted=false` 保留。

## 逐项结果

每个题目在eager/graph、attention DP0/DP1四种组合中各重复两次。temperature=0、seed=20261111、max_tokens=64，自然EOS，不修改原JSON提示、不使用约束生成。

| 题目 | eager DP0 | eager DP1 | graph DP0 | graph DP1 | 各组共同输出 |
| --- | --- | --- | --- | --- | --- |
| 原JSON排序 | 0/2 | 0/2 | 0/2 | 0/2 | `["json_array": [1, 2, 5, 9]]` |
| 纯数字排序控制题 | 2/2 | 2/2 | 2/2 | 2/2 | `1,2,5,9` |
| 12×13算术控制题 | 2/2 | 2/2 | 2/2 | 2/2 | `156` |

同题8次输出字节及SHA-256完全一致，JSON失败内容也与上一轮一致。所有回答自然结束、SSE完整且输出不足64 token，未被输出上限截断。纯数字控制题提示独立，不用于替代原JSON质量门槛；16/24是这三条诊断提示的通过数量，不能解释为通用模型准确率。

本批短题未表现出某一attention rank或graph模式特有的输出差异。graph日志有125条 `npu graph: True` decode记录，eager为0，证明图模式实际运行过图重放；计数包含启动warmup，不等于诊断请求数或单条请求的decode步数。

目前更应沿当前配置共有的计算路径查找差异，不能据此认定只是模型原生的JSON能力问题。此前注意力TP2/MoE EP2同一道JSON题通过；本轮没有与该配置进行固定token历史的logits比较，也没有独立全精度参考，因此attention TP变化、DP路径及相关数值/实现差异仍待区分。同题输出一致只覆盖这三条提示，不代表logits或完整模型数值等价。

## 专家、资源与运行

两个模型模式均通过86条完整attention TP1/DP2审计、86条EP2/TP1审计、两rank各1请求槽及HCCL AllReduce/scatter策略检查。NPU2完整专家0–31、NPU5完整专家32–63，CPU仅rank0执行64–255。候选源码四个文件的hash清单与上一轮一致；启动脚本只额外设置decode日志间隔1。

| 请求后快照 | NPU2 HBM | NPU5 HBM | 系统MemTotal−MemAvailable |
| --- | ---: | ---: | ---: |
| eager | 45.64 GiB（46733 MiB） | 45.58 GiB（46674 MiB） | 161.24 GiB |
| graph | 45.88 GiB（46982 MiB） | 45.82 GiB（46919 MiB） | 161.27 GiB |
| 此前attention TP2 / MoE EP2 graph | 42.80 GiB | 42.72 GiB | 160.75 GiB |

完整注意力副本使本轮graph整卡HBM相对此前每卡增加约3.1 GiB。以上为整卡观测值和整机内存压力，非单模块独占显存/RSS；CPU RAM权重没有裁剪。eager/graph快照的约0.24 GiB差值不能用来推断速度。

此次没有512 token正式性能请求，不能从2–17 token短题估计两模式的稳定decode性能或该配置的加速收益。

独立核对24条请求的矩阵身份、原提示、routed_dp_rank、seed/temperature、64 token上限、正文hash、chunk拼接、自然结束及时间戳；重新解析JSON和控制题结果，与自动汇总一致。启动时冻结脚本hash一致，本地/远端63项证据hash一致。源代码归档、完整源码目录和运行心跳不在证据文件对照范围，按既有采集策略保留远端。

HCCL三种规模求和及实际DP scatter小测通过；两rank均完成四种token分布的eager与两次NPU graph求和/分发检查。远端183条温度采样健康均OK，最高66°C/61°C。全部实验容器已停止，端口18082释放，原单卡服务保持停止；回读时两卡空载HBM3408/3407 MiB、健康OK且无计算进程。未更改工作站温控。

## 后续定位与证据

下一步有价值的是在固定token历史下比较attention TP1/DP2与此前attention TP2/DP1的logits，找到首次分歧，再按attention、MoE输入/输出及合并阶段缩小范围。原质量门槛继续保留，不据当前结果进入性能对照。本轮未自动启动后续实验、提交或发布。

[独立复核程序](verify_diagnostic.py)、[复核记录](runs/attn-tp1-ep2-20261124-214932-format-diagnostic/reviewed-diagnostic.json)、[原始24条请求](runs/attn-tp1-ep2-20261124-214932-format-diagnostic/remote/diagnostic-requests.jsonl)、[自动汇总](runs/attn-tp1-ep2-20261124-214932-format-diagnostic/remote/diagnostic-summary.json)、[证据hash](runs/attn-tp1-ep2-20261124-214932-format-diagnostic/remote-hash-check.json)。这些runs原始记录按项目惯例保留本地，不纳入Git。

```powershell
python reports/2026-11-24-taishan-attn-tp1-ep2/verify_diagnostic.py reports/2026-11-24-taishan-attn-tp1-ep2/runs/attn-tp1-ep2-20261124-214932-format-diagnostic --remote-check
```
