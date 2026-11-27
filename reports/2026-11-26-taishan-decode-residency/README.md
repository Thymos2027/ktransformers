# 910B decode 热专家驻留实验

2026-11-27回读：`decode-resident-20261126-232139-first`于00:23:36完成，24正式/8预热、140项远端证据及5份部署源码SHA核对通过。真实槽位/graph与decode更新跑通，但1K/4K decode分别慢13.31%/11.25%，总耗时增加10.81%/8.56%；每请求248次换入、约3.9秒更新，准备缓存命中为0。见[结果报告](REVIEW.md)。当前原型不建议默认开启，没有启动后续实验。

参考 vLLM-Ascend [#18604](https://github.com/vllm-project/vllm-ascend/pull/18604)，移植准备好的NZ物理字节缓存、固定槽位和完成后发布映射；新增有预算的decode热度更新，保留KT CPU专家计算。实验范围及限制见 [experiment-plan.md](experiment-plan.md)。

- `kt_decode_residency.py`：真实MXFP4转换、锁页主存缓存、ACL搬运、CPU/NPU掩码更新及graph回调观察。
- `residency_policy.py`：热度、预算和prepare/copy/fence/publish事务。
- `prepare_candidate.py`：只修改经SHA验证的隔离SGLang副本，记录全部部署源码SHA。
- `hardware_gate.py`：真实910B转换/拷贝/graph门槛及微基准。
- `run_guarded.py`：沿用现有隐藏后台观察、温控及NPU2隔离。
- `runs/<run-id>/status.json`、`remote-status.json`：本地和最近远端状态；结束后证据在 `remote/`。
- 远端 `controller.stdout.log`、`controller.stderr.log`、`logs/environment.log`、`logs/a0.log`等：控制器和各阶段日志。

自测：`python -m unittest discover -s reports/2026-11-26-taishan-decode-residency -p "test_*.py" -v`。

任务后台完成后由用户叫回，复核原始请求、执行日志、SHA和清理状态再报告收益。
