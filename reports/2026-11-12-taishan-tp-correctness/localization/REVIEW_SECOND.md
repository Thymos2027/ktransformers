# 局部数值定位第二轮：首个已采集 MoE 层出现差异

2026-11-15（北京时间）运行 `localize-20261115-second`，20:06–20:30。两组 eager 配置各完成 `prior_tp2`、`prior_tp1` 的两次固定历史请求及一条关闭 logits 采集的对照；结果已下载。原服务恢复后 capital、arithmetic、SSE 三项 smoke 均通过；随后只读复核仅原容器运行，`/health=200`。本机 154 个热采样无错误，最高 NPU 温度 60°C，风扇最低 6525 RPM。

第二轮按配置分隔层文件，共 16,512 份：TP1 rank 0、TP2 rank 0/1，各 43 层、64 次调用、归约前后各一份。远端原始时间将完整 prefill 的首个预测位置对应到 `prior_tp2` 的 call 012/023、`prior_tp1` 的 call 044/054；后一个数字是同配置重复请求。两次重复的比较指标完全相同。原始层文件与机器可读比较位于 `runs/localize-20261115-second/remote/layer-trace/`、`analysis-call012.json` 等文件。

| 输入 | 首个 MoE 层（layer 0）TP1 与 TP2 两 rank 求和后的归一化 RMSE / 最大绝对差 | layer 1 对应指标 |
| --- | ---: | ---: |
| `prior_tp2`，位置 0 | 0.014740 / 0.002930 | 0.046597 / 0.006470 |
| `prior_tp1`，位置 0 | 0.015159 / 0.002747 | 0.044992 / 0.007446 |

因此**最早被本轮探针观察到的跨 TP 差异在 layer 0 的路由专家输出**，其后差异增大。两个输入的完整词表 logits 也重现了不同 top-1：`prior_tp2` 为 token 2337/9544，`prior_tp1` 为 9544/2337（TP1/TP2）。这并不证明差异起于 layer 0；该层输入已经过 attention、归一化和路由，仍可能更早发生分歧。

解释边界：这轮插桩位于 `FusedMoE.forward_impl` 的 `dispatcher.combine` 后、`DeepseekV2MoE.forward_normal` 的缩放、共享专家合并及 TP all-reduce 前。所查四次调用中，TP2 两 rank 的该插桩点“pre/post”各自逐位相同，说明 `FusedMoE` 内的可选归约未在这些调用上执行；两个 rank 的张量是局部贡献，不能直接拿 rank 0 与 TP1 作逻辑输出比较。表中以 FP32 将 TP2 两 rank 的局部输出相加，仅作定位近似，不等于实际 BF16 集体归约，也不能据此证明 CPU 输出恰好加入一次。

下一步应在 layer 0 同一位置采集 MoE 输入、路由 ID/权重、NPU 专家输出、CPU 专家输出及真实 TP 归约前后张量，并核对输入是否已分歧。若输入相同，再分别用同一冻结激活检验分片权重/scale 与 CPU 合并；若输入不同，先回溯 layer 0 的 attention 和归一化。当前没有数值验收标准或任务质量结论，TP2 正确性仍未验收。
