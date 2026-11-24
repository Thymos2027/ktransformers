# 首轮：DP请求池被全局上限1整除为0

运行 `attn-tp1-ep2-20261124-203552-first` 于2026-11-24失败。环境、真实镜像默认设备检查、HCCL通过。两rank权重均加载完毕，86条完整attention TP1/DP2审计和86条专家EP2/TP1审计完整；首层注意力完整64 heads，wq_b=[32768,1024]、wo_b=[4096,8192]。专家按0–31/32–63分配，CPU仅rank0。

直接错误为 `AssertionError: max_running_request is zero`，发生在eager启动的请求池分配阶段，没有完成服务健康、短题或性能请求。日志两rank均有 `Initialize DeepSeekV4TokenToKVPool with max_num_reqs=0`。

固定源码 `model_executor/pool_configurator.py` 的DSV4配置器计算 `requested_max_running_requests_per_worker = max_running_requests // attn_dp_size`。启动器沿用全局max-running-requests=1，DP=2后结果为0；此前计划将它理解成每rank上限，这是参数解释错误。

修正为全局 `--max-running-requests 2`，使每个attention DP worker上限为1；客户端仍顺序发送请求，不开始并发压测。新增真实镜像环境探针 `request_budget_probe.py`，解析实际隔离源码中的赋值表达式，验证旧配置1//2为0、当前2//2为1，再允许模型加载。启动验收必须读取到两rank各1个请求槽。

本地17项检查通过，包括零槽拒绝、启动参数匹配及真实固定源码公式；远端缺少源码快照的检查明确跳过，实际镜像环境probe另执行。新轮完整重跑，不复用失败模型阶段。

原轮cleanup通过，容器退出1、OOMKilled=false、PID0，原服务保持停止、18082释放。完整模型加载后仍有约19GB可用HBM，本次失败由请求上限参数触发，不是OOM或温度告警。既有温控策略和源码隔离范围保持。
