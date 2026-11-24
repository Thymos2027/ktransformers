# DS-V4-Flash：两卡不同专家的 EP2 实验

2026-11-24，用户已授权尝试 NPU2/NPU5 各放不同的 32 个专家。目标：attention TP2、MoE EP2/TP1；物理 NPU2（rank0）完整专家0–31，NPU5（rank1）完整专家32–63，CPU executor 仅 rank0、使用全局256专家的 mask 跳过两卡并集0–63，计算其余64–255。RAM存储精简不在本轮范围。

复用上轮固定 Pan-Boyi SGLang ff46d660d16cbf11ef393c80cc6b3ab7f6259780、匹配KT和W8A8/MXFP4权重、镜像 f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e。仅隔离源码修改：opt-in EP2门禁、rank-local mask/slot、保持全局专家ID至KT loader、CPU并集mask。保持默认TP1拒绝非授权配置，禁用EPLB和非前缀放置。每卡实际路由专家张量应为32个完整专家，中间维度2048，不是64个半专家。

本次为复制token输入后按专家归属本地筛选、最终AllReduce求和的EP2路径；保留AscendTPDispatcher，不实现AllToAll。注意力仍TP2。两rank收到相同全局路由，每rank只处理所属常驻专家；rank0一次加入CPU补集。全局专家数仍256，不能直接改为64。检查86条逐层/rank驻留审计、实际张量形状、CPU唯一所有者以及身份expert mapping。

首轮因新增CPU表构造未显式指定device，在模型默认NPU设备上下文中触发CPU/NPU比较错误，详见REVIEW_FIRST.md。修复后新增CPU表均明确device=cpu，真实镜像环境阶段执行device_probe.py：重现历史失败、再在非CPU默认设备下验证两个rank的实际生成构造函数及identity映射门禁，通过后继续。该probe不启动完整模型或代替实机数值验收。

第二轮已通过环境、HCCL、eager三题与86条审计，图模式也完成双rank捕获，但旧TP日志检查未匹配TP/EP组合标签，详见REVIEW_DEVICE_FIX.md。修正标签解析并用真实日志验证后，在新目录运行 `ep2-expert64-20261124-graph-resume`；通过完整实现、原生环境、权重、参数、eager原始日志和短题一致性校验后，仅复用第二轮eager证据，图模式捕获、五题与性能全部重新运行，保留来源哈希。

顺序：本地/远端CPU边界及布局测试；环境导入和HCCL；EP2 eager三条短题；EP2 decode图捕获和五题短检查；全部通过后执行144/792输入、固定512输出，每条预热一次、正式三次，共6条正式和2条预热。性能为探索性，后续与旧TP2/K64比较需说明跨运行和生成路由差异。短题、数学布局测试、驻留检查不能替代独立的实机全logits/任务质量验收；遇错误立即停止，不回退为相同专家或重复CPU结果来凑通过。

单请求，96 CPU线程/4池，max-total-tokens16384、mem-fraction0.81、chunked-prefill512，其他参数沿用上轮。新增 KT_NPU_EXPERIMENTAL_EP2=1，CLI --tensor-parallel-size 2 --expert-parallel-size 2 --kt-num-gpu-experts 64；这里64为两卡常驻总数，局部32在隔离wrapper中派生。

后台运行：现有Taishan-NpuCooling负责风扇，本轮只读状态；>=90°C连续10秒停，真实遥测/健康及90秒远端心跳保护保留，SSH75秒内重试。仅本轮新容器、端口127.0.0.1:18082；原服务已停止，结束保持原状态。完整stdout/stderr、status、commands、热采样、布局和资源证据留存；后台启动后短查即可，用户再次叫回才复核结果。不上线、不发布、不自动commit。
