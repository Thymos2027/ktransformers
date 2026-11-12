# TP1/TP2 实验的社区分享价值与 GitHub 入口

2026-11-12 更新：已按用户分享意图，以 `Thymos2027` 在 [KT #2157 原线程发布英文实验记录](https://github.com/kvcache-ai/ktransformers/pull/2157#issuecomment-6810709188)，发布后回读确认正文与[本地发布稿](COMMUNITY_COMMENT.md)一致。内容包括固定版本、功能验证、KV 上限 workaround、性能与限制，以及折叠的完整候选 diff、启动参数、输入构造、24 条逐次测量和输出哈希。未创建新 issue/PR；服务器配置未改动。GitHub 连接器写入返回 403 后，使用本机已登录的 GitHub CLI 完成发布。

核查日期：2026-11-11（北京时间）。本次通过 GitHub 连接器只读查询 issue、PR、讨论评论及当前源码/文档，没有发送评论、创建 issue/PR 或发布文件。研究结果见[速度报告](REVIEW.md)和[TP2 功能报告](../2026-11-10-taishan-tp2/REVIEW.md)。以下取舍是对已查证来源的判断，不表示维护者已经接受该贡献。

## 结论

已有可分享价值，适合定位为 **Kunpeng 920 + 两张 910B4-1 的 KT 专家卸载实验验证与部署边界**。最有用的是三组证据：受保护的 TP2 功能运行、自动 KV 预算 OOM 及显式缓存上限的恢复方法、固定专家集合下 TP1/TP2 速度仅小幅变化。现在可交付精简验证报告或原讨论中的补充评论；尚不足以直接提交取消 NPU TP 限制的通用支持 PR，或以 2%–3% 为标题宣传已证明的性能优化。

## 已查证的上游入口

| GitHub 来源 | 核查结果 | 与本实验的关系与建议 |
| --- | --- | --- |
| [KT #2157：Ascend NPU backend](https://github.com/kvcache-ai/ktransformers/pull/2157) | 已合并；作者 Pan-Boyi。范围为单卡，明确把多卡卸载、TP/EP 和并发留作后续工作 | 最直接的代码背景。适合向原作者补充 TP2 的硬件验证、实际补丁范围、限制和证据链接，询问未来支持所需门槛 |
| [KT #2159：单卡教程](https://github.com/kvcache-ai/ktransformers/pull/2159)及[当前教程限制](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md#known-limitations) | PR 已合并；当前教程仍限定 TP=EP=1、单请求。PR 讨论中维护者曾要求精简大量脚本 | 可以形成小篇幅的实验性双卡附注或独立验证报告；不把整套服务器/风扇/恢复控制器直接倾倒进主线，也不直接删除 Known Limitations |
| [Pan-Boyi/sglang 的 KT wrapper](https://github.com/Pan-Boyi/sglang/blob/dsv4-cann9-no-patch/python/sglang/srt/layers/moe/kt_ep_wrapper.py) | 查询时仍显式拒绝 NPU + KT 的 TP≠1 或 EP≠1 | 我们测的是固定 fork 上的显式 opt-in 候选。分享时必须公开基线 SHA 和 candidate diff，不能称官方 SGLang 已支持此路径 |
| [SGLang #34673：Restore KT CPU-expert offload on Ascend](https://github.com/sgl-project/sglang/pull/34673) | 关闭、未合并、Draft；作者 Nikonz。机器人说明因 WIP 长期未更新而关闭，允许相关工作继续时重开 | 说明上游整合仍有缺口。适合提供硬件验证材料帮助后续恢复工作；当前速度不是该 PR 或 SGLang main 的基准结果 |
| [SGLang #23598 的 TP2/4 用户反馈](https://github.com/sgl-project/sglang/issues/23598#issuecomment-4548249592) | 用户报告 TP1 可跑，而 TP2/4 遇到 SparseAttnSharedkvMetadata 错误；该大讨论已关闭 | 有直接的少卡运行需求。不过对方配置不是本次已复现的 KT 卸载路径；只能作为相关案例交流，不能回复“这个方案修好了你的 bug” |
| [KT #1882：Ascend + Kunpeng 请求](https://github.com/kvcache-ai/ktransformers/issues/1882) | 仍开放；2026 年 3 月的旧回复称完整 A+K 支持无时间表，后续 #2157/#2159 已合并 | 平台层面可补一条简短的支持状态更新与现有 PR 链接；TP2 性能主帖不应重复灌入多个老讨论 |

另核查了 [SGLang #41163](https://github.com/sgl-project/sglang/pull/41163)：它修复的是 NIXL PD 描述符与 c4 indexer dummy page 数量不一致。它不能作为我们 NPU KV 自动预算 OOM 的同一根因或已有修复证据。对本次查询覆盖到的讨论，未找到可直接等同于我们的固定 fork、TP2、KT 卸载 OOM 复现；搜索覆盖有限，不据此声称发现新 bug 或首次支持 TP2。

## 能够分享的三项结果

1. **TP2 的实验性运行证据。** 一个模型、两个 TP rank、一个 API，eager 和 batch=1 decode 图都通过有限短任务；两卡 HCCL 小张量求和通过。代码贡献是精确限制范围的 opt-in 门禁与诊断、实验控制和验证；CPU wrapper、专家分片、通信与合并的核心实现属于原项目作者。还不能把删除限制包装成原创 TP 实现。
2. **可重现的缓存配置问题。** 两卡各约 21.9 GiB 权重加载完成后，自动池预算到约 376 万 token，在 NPU indexer 池分配 OOM；显式 `--max-total-tokens 16384` 后同路径可启动。它能帮助同环境用户避坑，也提供后续内存预算核算的复现入口。根因与通用修复未定位，当前证据只支持固定版本的观察及 workaround。
3. **双卡速度收益的有界数据。** 固定 32 常驻专家、96 CPU 线程、相同缓存上限，ABBA 四次启动、24 条正式请求、144/792 输入和 512 输出。TP2 decode 中位数提高 3.37%/2.23%，总耗时少 2.87%/1.55%，TTFT 无改善。对这组 CPU 卸载配置，直接增为两卡没有获得大幅速度改善。这是有价值的部署取舍数据，不是一般 NPU TP 缩放规律。

## 发布前怎样压缩与补强

现在就能整理一份英文短报告：硬件及 SYS 拓扑、固定提交/镜像/权重身份、候选 diff、两条启动命令、三项运行门槛、OOM/缓存上限、两行性能表、原始逐次 JSON 和限制。服务器 IP、SSH 别名、iBMC 控制及环境私有路径不属于公开复现所需信息，公开副本应参数化。大日志保留本地，提供精简相关堆栈、必要 capture/分片日志和 SHA-256；公开证据链接需使用已发布的固定 commit，不能把本地路径当作其他人可访问的证明。

原文标题可采用：**Experimental DeepSeek-V4-Flash CPU offload on Kunpeng 920 + 2× Ascend 910B: TP1/TP2 validation, KV sizing, and limited scaling**。

若目标升级为默认启用 TP2 或代码支持 PR，需要补足固定历史/层输出/logits 的数值检查和代表性任务准确率，核查 CPU 专家输出只合并一次、W8A8 分片与缩放的正确性；当前长文本跨 TP 不同，不能只以温度 0 或五条短任务作为等价证据。OOM 修复 PR 还需要定位预算缺项并在相应目标上游版本重现，不能把工作分支问题直接报成 SGLang main 回归。更广泛性能声明需要更多独立启动/请求分布，且区分固定专家数与增加常驻专家数的实验。

建议顺序：先在自己的研究目录发布精简且可复核的英文实验报告；取得明确外发授权后，优先在 KT #2157 的原线程向原作者提供一次简短验证补充。根据作者反馈决定是文档附注、独立讨论，还是补齐门槛后提交代码。当前不需要为了分享把现有基准继续扩成无限实验，也不重复向多个无关路线图发送相同数据。
