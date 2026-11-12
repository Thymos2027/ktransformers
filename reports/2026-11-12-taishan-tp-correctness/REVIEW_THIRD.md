# 第三次短探针失败复盘

2026-11-12（北京时间）运行 `correctness-20261112-pilot3` 在首个 `tp1_eager`、`chinese` 请求末尾返回 HTTP 500。与前两次不同，归档的 `probe/rank0.jsonl` 已记录位置 0–8 共 9 个目标前向：位置 0 为 `EXTEND`，位置 1–8 为 `DECODE`，每步 decode 输入与冻结 token 前缀一致。采集器随后遇到位置 9，超过 8 个冻结后续 token，主动抛出 `KT probe position exceeds frozen sequence`。

请求设置 `max_new_tokens=9`，固定 fork 的 overlap 调度器可能在最终 token 的完成状态处理之前多发出一个前向；这与源码执行顺序和本次位置 9 观察相符。位置 9 已超出预定数值测量范围，不能当作第十个正式比较位置。采集器现单独记录该额外前向，不保存其 logits；客户端仍严格验收目标位置 0–8，并限定额外前向最多一次且只发生在位置 9。

本地模拟覆盖这种额外前向，采集器测试、`compileall` 与 `git diff --check` 均通过。第三次运行目录保留；报告 `original_restored=true`，后续只读核查仅原单卡容器运行且健康接口返回 200。本次仅有部分 TP1 数据，没有跨 TP 正确性结论。
