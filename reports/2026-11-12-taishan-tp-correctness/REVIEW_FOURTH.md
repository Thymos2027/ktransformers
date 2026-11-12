# 第四次短探针失败复盘

2026-11-12（北京时间）运行 `correctness-20261112-pilot4` 完成 TP1 eager 前 7 条输入、共 14 个正式采集请求，目标位置 0–8 的本地结构复核全部通过。第 8 条 `prior_tp2` 使用 792-token 提示词，超过固定的 512-token 分块 prefill 大小。采集器在未完成提示词的中间块看到 `seq_len < prompt_len`，错误地把它当作无效生成位置，抛出 `KT probe position precedes prompt`，使请求返回 HTTP 500。

现将中间 prefill 块单独记录到 `rankN-partial.jsonl`，仅在完整提示词完成后开始记录位置 0，再记录冻结 8-token 历史下的位置 1–8。客户端校验中间块长度必须小于完整提示词长度，且模式为 `EXTEND` 或 `MIXED`。本地模拟包含 792-token 提示词的 512-token 中间块、完整 prefill 与 decode；测试通过。下一次先只运行这条长提示词的 TP1 eager 窄范围检查，通过并复核后再重跑四种配置；正式运行也将长提示词排在前面，并在配置中记录执行顺序。

第四次失败前已有的 TP1 eager 数据保留为诊断证据；没有完整跨 TP 对照或正确性结论。流程报告 `original_restored=true`，后续只读核查仅原单卡容器运行且健康接口返回 200。
