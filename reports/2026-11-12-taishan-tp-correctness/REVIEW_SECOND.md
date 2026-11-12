# 第二次短探针失败复盘

2026-11-12（北京时间）运行 `correctness-20261112-pilot2` 仍在首个 `tp1_eager`、`chinese` 请求处失败。日志明确记录 `RuntimeError: Duplicate KT probe capture: /experiment/probe/tp1_eager-chinese-r0-rank0-step000.pt`；首次回调已成功保存 129280 维 FP32 logits 和 `EXTEND` 元数据。第二次回调因采集器用 `len(req.output_ids)` 作为模型步骤，重复使用 `step000` 文件名，随后请求返回 HTTP 500。原服务自动恢复；只读核查仅原容器运行且 `/health` 返回 200。

固定 fork 的 overlap 调度器在当前批前向之后才处理上一批结果，`req.output_ids` 的更新可能落后于已经推进的模型位置。这个执行顺序可解释重复的 `step000`；第二次回调的实际位置在原采集器抛错前未记录，因此此点仍属源码支持的推断，不能称为已实测证明。

采集器现以实际 `seq_len - prompt_len` 编号，并在 decode 时核对真实输入 token 与冻结历史；同时记录调度器已发布的历史长度。若同位置确实重复执行，采集器保留独立的 `callN` 文件与记录，让客户端报告明确的结构检查失败，避免因文件重名使调度器崩溃。本地模拟覆盖 `array` 历史滞后和同位置重复回调，测试通过；身份校验后的候选补丁在隔离源码副本中编译通过。第二次运行仍没有 TP 正确性结论。
