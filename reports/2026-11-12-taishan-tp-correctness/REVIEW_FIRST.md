# 首次短探针失败复盘

2026-11-12（北京时间）运行 `correctness-20261112-pilot` 在首个 `tp1_eager`、`chinese` 请求处失败。远端 `/generate` 返回 HTTP 500；[本机归档日志](runs/correctness-20261112-pilot/remote/logs/tp1_eager.log)显示调度器调用 `KTFrozenHistoryProbe.__call__` 时，`json.dumps(req.output_ids)` 抛出 `TypeError: Object of type array is not JSON serializable`。该字段是 SGLang 的 `array`，并非普通 Python `list`。失败发生在采集器写历史哈希阶段，尚无可用数值对照，不能据此判断 TP 正确性。

已改为 `json.dumps(list(req.output_ids))`，并加入模拟 `array` 历史、逐步强制 token、保存与关闭保存的本地测试。`python reports/2026-11-12-taishan-tp-correctness/test_probe_fragment.py` 通过，`compileall` 与 `git diff --check` 通过。原失败运行目录和远端目录均保留，不覆盖原始证据。

失败流程报告 `original_restored=true`；随后只读核查仅原容器 `dsv4-kt-20261104` 运行，`http://192.168.137.11:18080/health` 返回 200。无 NPU 温度或风扇告警。
