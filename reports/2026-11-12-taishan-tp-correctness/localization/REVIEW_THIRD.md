# Layer 0 组件定位：差异已在 MoE 输入处存在

运行 `localize-20261116-layer0`，2026-11-16 08:35–08:58（北京时间）。TP1/TP2 eager 均完成 `prior_tp2`、`prior_tp1` 的两次固定历史请求和关闭 logits 采集的对照。结果已下载；状态为 `complete`、`error=null`、`original_restored=true`。恢复后的三项 smoke 通过，随后只读复核仅原容器运行且 `/health=200`。本机 151 个热采样无错误，最高 NPU 温度 58°C，风扇最低 7350 RPM。

本轮只为 layer 0 保存 `moe_input`、KT dispatcher 前后、`DeepseekV2MoE` 实际 all-reduce 前后的末位置张量。两配置分目录保存：TP1 rank 0 有 320 份，TP2 rank 0/1 共 640 份。远端原始时间将两个提示词的首个预测位置对应到 call 012/023、044/054；后者为重复请求。逐项机器可读结果在 `runs/localize-20261116-layer0/components-comparison.json`。

| 首个预测位置 | 进入 MoE 前 TP1/TP2 激活差异（归一化 RMSE / 最大绝对差） | CPU 专家输出差异（归一化 RMSE） | 真实归约后 layer 0 输出差异（归一化 RMSE） |
| --- | ---: | ---: | ---: |
| `prior_tp2` | 0.005349 / 0.0078125 | 0.014740 | 0.027032 |
| `prior_tp1` | 0.005392 / 0.0058594 | 0.015159 | 0.028097 |

两次重复的指标完全一致。TP2 两 rank 的 MoE 输入、路由 ID 和权重彼此逐位相同；与 TP1 相比，输入和路由权重已不同，但六个 top-k 专家 ID 均相同，依次为 107、85、126、215、44、129，全部在 0–31 常驻范围之外。这四个调用的 NPU 专家输出在 TP1 与 TP2 两 rank 上均为全零。CPU 专家输出在 TP1 和 TP2 rank 0 非零，TP2 rank 1 为全零。因而本轮首位置的专家输出差异不能归因于 NPU 常驻专家分片；输入差异已发生在专家计算和 KT CPU 合并之前。

四个调用均记录 TP1 不执行、TP2 执行 `DeepseekV2MoE` 的 TP all-reduce；TP2 归约后两个 rank 的 layer 0 输出逐位相同，并与两个 rank 的归约前 BF16 张量之和逐位相同。结合 KT hook 中 rank 0 的一次 CPU 加入、rank 1 的零 CPU 输出，**这些调用未显示 CPU 输出重复归约或缺失**。此结论只覆盖所采集的 layer 0、首个预测位置和全 CPU 路由样本，不能推广到混合路由或其他层。

下一步应在 layer 0 的 MoE 输入之前回溯：用相同 token/位置核对 decoder 输入、attention 前归一化、attention 输出及 MoE 前归一化，找出最早分歧点；若更早的输入一致，再查 attention TP 分片、权重/scale 与归约。当前 TP1/TP2 的首位置 logits 仍不同，没有冻结数值容差或独立质量评测，TP2 正确性尚未验收。
