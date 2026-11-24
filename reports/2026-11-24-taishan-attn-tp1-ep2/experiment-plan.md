# 注意力TP1、MoE EP2/TP1实验

用户于2026-11-24授权。在物理NPU2和NPU5上，两卡各有完整注意力副本；单请求归属其中一个attention DP rank，进入MoE前聚合token，MoE分别执行完整专家0–31/32–63，最终求和并将结果送回请求所属rank。CPU仅rank0执行专家64–255，RAM权重不裁剪。

启动参数 `--tp 2 --dp 2 --enable-dp-attention --ep 2 --moe-a2a-backend none`。这里CLI TP是两进程组，实际attention TP=2/2=1，attention DP=2，MoE EP=2、TP=1。逐层43×2条注意力审计必须证明实际attn_tp_size=1、完整heads和groups；另保留86条专家驻留、完整维度和CPU唯一所有者审计。禁用CP、gatherv、reduce-scatter替换和共享专家本地优化，使用原有gather/AllReduce/scatter路径。

第二轮发现EP2/DP2独立策略仍自动启用依赖PyNCCL的reduce_scatterv，见REVIEW_REQUEST_POOL_FIX.md。本实验新增NPU opt-in策略门禁，确保MoE内部恢复一次HCCL AllReduce、decoder执行DP scatter，逐层审计合并路径。HCCL阶段增加真实BF16求和/DP分发与NPU图重放小测，覆盖空闲rank和不等长token；通过后才加载模型。不把通信probe视为完整数值验收。

复用固定SGLang ff46d660d16cbf11ef393c80cc6b3ab7f6259780、镜像f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e和W8A8/MXFP4权重。只修改新目录隔离源码，不改原服务、权重、镜像和工作站温控。完整注意力副本可能增加每卡HBM，启动失败即停并清理，不自动调整专家数。

顺序：本地及远端布局/图标签/拓扑检查、真实镜像默认设备探针、HCCL、eager三题、graph双rank捕获及五题，再进行144/792输入、固定512输出，各预热一次、正式三次。所有阶段重跑，不复用旧eager或性能结果。此次不进行eager与graph同负载速度对照，完整数值和任务质量仍不算验收。

96 CPU线程/4池、单请求逐条发送，CLI全局max-running-requests=2，经attention DP=2划分后每rank为1，mem-fraction=0.81、max-total-tokens=16384、context-length8192。首轮误传全局1导致两rank请求槽均0，见REVIEW_FIRST.md；新轮先用实际DSV4源码公式进行环境探针，再核对两rank真实请求槽均1。DP attention会将CLI chunked-prefill-size除以2，故传1024以保持实际每rank512；该参数差异和attention复制均须在对照报告中说明。LM head采用默认路径，不额外启用DP LM head。

现有Taishan-NpuCooling只读，实验>=90°C持续10秒停止，75秒SSH恢复预算及90秒远端心跳保持；端口18082、原服务初始/最终均停止。后台任务脱离终端，保留状态、stdout/stderr、采样及退出状态，短查启动后结束本轮，用户叫回再复核。不发布、不自动commit。

第三轮模型和通信路径通过，但graph第五题输出非法JSON，见REVIEW_HCCL_COMBINE_FIX.md。后续仅运行 `--diagnose-format` 定点诊断：原JSON提示不变，额外纯数字排序/算术控制题，固定DP0/1，eager/graph各重复两次，共24条短请求。max_tokens64、自然EOS，decode日志间隔1；不执行512 token性能矩阵、不放宽原验收条件。答案失败记录后继续收集诊断矩阵，传输和运行异常仍失败即停。诊断完成不代表模型质量验收。
