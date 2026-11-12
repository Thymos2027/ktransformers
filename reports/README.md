# taishan 研究与实验报告

更新：2026-11-11（北京时间）。本目录记录鲲鹏 920 + Ascend 910B 上的 KTransformers（KT）与 SGLang 联合验证。各报告中的“当前”只代表记录时的状态；服务是否仍在运行需要另行核查。

## 阅读入口

1. [研究计划](RESEARCH_PLAN.md)：当前单模型双 NPU TP=2 目标、实验范围与验收门槛。
2. [研究状态](RESEARCH_STATUS.md)：已完成、待验证及当前下一步；不代表服务器实时状态。
3. [阶段成果](RESEARCH_OUTCOMES.md)：可核对的发布、部署和离线研究结果及其限制。

## 原始报告

| 日期 | 报告 | 已验证的范围 |
| --- | --- | --- |
| 2026-11-04 | [DeepSeek-V4-Flash 部署与验收](2026-11-04-taishan-deepseek-v4-flash/README.md) | 单张 910B4-1、单请求、8192 token；图捕获、健康检查、短聊天与流式结束通过 |
| 2026-11-04 | [逐层 MXFP4 CPU 专家发布与审计](2026-11-04-taishan-deepseek-v4-flash/community-release/README.md) | 43 个 GGUF 完整审计、ModelScope 发布和远端大小/哈希核对；KT 文档 PR #2257 后续已合并 |
| 2026-11-07 | [专家路由与缓存策略研究](2026-11-07-taishan-expert-cache-study/README.md) | 20 条受控请求的路由采集、离线策略回放和单专家搬运微基准；未实测策略端到端加速 |
| 2026-11-08 | [固定单卡基线首轮实验](2026-11-08-taishan-single-card-baseline/REVIEW.md) | 三条正确性探针和九条正式顺序请求通过；记录 TTFT、decode 速度和资源快照，未测持续吞吐或并发 |
| 2026-11-08 | [双卡双实例对照](2026-11-08-taishan-dual-instance/REVIEW.md) | 两条同时到达请求：单实例 13.55、双实例 2.44 tok/s；第二实例已停止，退化根因未定 |
| 2026-11-08 | [双实例退化分层诊断](2026-11-08-taishan-dual-instance-diagnosis/REVIEW.md) | 仅两实例同时执行时均降至约 1.25 tok/s；CPU 压力高、无活跃换页；已清理第二实例 |
| 2026-11-08–10 | [双实例线程预算与结项](2026-11-08-taishan-dual-instance-thread-budget/CLOSEOUT.md) | 96+32 受控重跑完成，未解决并发退化；系列结项，原单卡服务已恢复 |
| 2026-11-10 | [重启后恢复与高温核查](2026-11-10-taishan-recovery/README.md) | 高温及掉卡原因未明；恢复后受保护重跑最高 61°C/60°C，21:25 原单卡服务健康，第二实例停止 |
| 2026-11-10–11 | [单模型双 NPU TP=2 验证](2026-11-10-taishan-tp2/REVIEW.md) | 限制缓存容量后，HCCL、eager 和 decode 图模式短测通过；原单卡服务已恢复。未测 TP 性能或完整数值等价 |
| 2026-11-11 | [TP=1 / TP=2 速度对照](2026-11-11-taishan-tp-speed/REVIEW.md) | ABBA 24 条正式请求通过；TP2 decode 高 3.37%/2.23%，总时长少 2.87%/1.55%。跨 TP 输出不同，结论限同输入同长度负载；原服务恢复 |

权重、构建产物和大体积运行日志不纳入本目录。新实验按运行开始日建独立目录，保留脚本、固定环境、原始逐次结果、失败记录及结果报告；根目录三份研究文档只维护当前判断和证据入口。
