# 第三轮：部署与通信通过，排序题JSON格式失败

运行 `attn-tp1-ep2-20261124-211254-hccl-combine-fix` 的最终状态为failed，cleanup_verified=true。直接原因是 `ep2graph quality gate failed: sorting`；不是前两轮的请求池或NCCL调用错误。

本轮实际双卡BF16合并probe全部通过：两rank均完成[3,0]、[0,3]、[2,3]、[1,1] token分布的eager与两次NPU graph重放，CPU贡献仅rank0加入一次。HCCL三种规模求和同样通过。

eager与graph各有完整86条注意力TP1/DP2和86条EP2驻留审计，HCCL AllReduce/scatter策略审计完整，请求池均为每rank1。eager模型三题巴黎、42、READY全部正确。graph模型两个rank图捕获成功、健康检查通过；前四题分别回答巴黎、42、156、READY，正常结束。

第五题原始提示为 `Return only a JSON array sorting [9,1,5,2] in ascending order.`，完整响应如下：

```text
["json_array": [1, 2, 5, 9]]
```

数字次序正确，但它不是合法JSON数组。该响应输出17 token、finish_reason=stop、SSE完整，未碰到64 token上限；检查器拒绝它是正确行为，不能忽略包装、提取内层数组或修改原题来将本轮标为通过。没有开始512 token性能请求，本轮不存在可比较的正式decode速度。

现有证据不能确定这是attention rank差异、eager/graph差异、配置数值变化还是指令遵循问题。既有短输出日志间隔较长，完整模型capture成功也不足以单独量化这五题的实际graph重放次数。因此新增独立短输出诊断，不修改原质量门槛或原失败状态。

诊断完整重跑环境、通信probe和两个模型模式；使用原JSON排序提示、额外的纯数字排序及算术控制题，固定 `routed_dp_rank` 0/1，每项重复两次，共24条，temperature=0、seed=20261111、max_tokens=64、自然EOS。模型参数与候选源码保持第三轮配置，仅decode日志间隔改为1以记录是否重放。回答失败作为诊断观察保留，传输/运行异常仍立即停止；不采集性能、不自动认可原质量门槛。

诊断的complete只表示收集矩阵完成，必须同时查看 `diagnostic-summary.json` 的逐项答案、passed、rank与模式对照，`original_quality_gate_accepted=false` 和 `correctness_accepted=false` 始终保留。本地27项检查通过，含非法JSON仍被拒绝、原提示不变、请求显式路由及采样参数保持。

第三轮已清理全部实验容器、释放18082，原服务保持停止；原日志及结果保留在对应runs目录，不重命名或覆盖。
