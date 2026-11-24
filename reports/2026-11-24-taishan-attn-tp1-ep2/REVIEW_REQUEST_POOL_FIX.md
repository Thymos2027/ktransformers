# 第二轮：NPU误入依赖PyNCCL的reduce_scatterv合并路径

运行 `attn-tp1-ep2-20261124-204735-request-pool-fix`。请求池修复有效：两rank均创建max_num_reqs=1，完成权重加载、86条attention TP1/DP2和86条EP2驻留审计，以及HCCL DP预热。之后在服务warmup的首次eager prefill失败，尚未通过健康检查或进入实验短题/性能采集。

直接堆栈为DeepseekV4的 `_run_moe_ffn_dp_sync` 调用 `get_tp_group().reduce_scatterv`，进入parallel_state.py第1177行 `with pynccl_comm.change_state(enable=True)`，而NPU的PyNCCL通信器为None，报 `AttributeError: 'NoneType' object has no attribute 'change_state'`。其后的HTTP warmup 600秒超时是scheduler退出的后果。

根因是 `layers/moe/utils.py::should_use_dp_reduce_scatterv` 对attention DP2、EP2、总组2自动返回True；此前关闭的 `SGLANG_DP_USE_GATHERV` 和 `SGLANG_DP_USE_REDUCE_SCATTER` 不控制这条独立规则。它也使DeepseekV2MoE跳过原有post-experts AllReduce，因此不能仅跳过崩溃调用而不恢复求和。

修复仅在新隔离源码且 `_is_npu` 和 `KT_NPU_EXPERIMENTAL_ATTN_TP1=1` 同时成立时，使这条共享策略返回False。这样MoE恢复一次HCCL AllReduce，decoder使用既有DP scatter取回本rank的token；两个决策读取同一函数，避免漏求和或重复求和。不改通用reduce_scatterv接口、不引入NCCL模拟、不改变专家分工或CPU唯一所有者。

本地24项检查通过，含固定源码旧路径重现、修复后两个gate一致、非NPU/非本实验配置保持原策略、已有融合求和的skip规则保持。新HCCL阶段在真实镜像上解析实际策略，调用HCCL求和与实际DP scatter函数/内存kernel，覆盖token数[3,0]、[0,3]、[2,3]、[1,1]，核对仅rank0加入一次CPU贡献的BF16结果，分别检查eager及两次NPU graph重放。该probe不是完整MoE算子或logits验收；通过后才加载完整模型。

新轮还会在86条attention审计中明确记录 `combine=hccl_allreduce_scatter`，不复用旧轮模型阶段。请求上限仍全局2、每DP rank1，客户端仍顺序发送。原失败状态及日志保持。

本轮cleanup_verified=true，原服务保持停止。失败依据为明确的通信路径异常，温控策略没有触发该错误；工作站风扇服务及90°C持续10秒停止策略不变。
