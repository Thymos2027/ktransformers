# 首轮：启动检查错误，尚未进行性能测量

2026-11-26，北京时间。运行 `tutorial-20261126-110255-first`。远端控制器以 failed 结束，本地证据已回传。

直接失败为 `Missing tutorial option/graph: swa_full_tokens_ratio=0.5`。模型加载成功，03:07:26 UTC 完成decode图捕获，03:07:33服务ready；控制器在启动参数审计处拒绝日志，尚未冻结输入或发送正式/预热性能请求。03:07:38的SIGTERM是失败收尾发出的停止指令，后续CancelledError不是首发故障。未见启动OOM，容器OOMKilled=false，清理验证通过，两卡无计算进程。

根因是本轮控制器错误沿用旧实验显式设置的0.5作为默认值。固定SGLang ff46d660的 `python/sglang/srt/arg_groups/overrides.py:1127` 实际为DeepSeek-V4设置默认0.1，首轮resolved server_args和模型覆盖日志均确认0.1。教程table12/table39不传SWA比例时应接受该模型默认；table19仍必须显式0.35。修复只改变审计预期，不修改模型默认或启动计算参数。

同时检查后续尚未执行的接口发现：该分支 `/tokenize` 要求 `prompt` 或 `messages`，不能沿用 `/generate` 的 `text` 字段。修正原始文本预分词请求为 `prompt`，避免下一阶段422，性能生成仍使用公开脚本的text请求。原始输入不套chat模板。

回归保存首轮实际启动日志的4行配置/图捕获/streaming slot为 `startup-reference.txt`，检查模型默认、16K显式覆盖、32K专家/内存配置、错误比例、静默回退和KV上限偏离不能通过；补分词接口字段回归。源码哈希清单仅包含静态源文件，排除运行中变化的status等JSON。

新运行使用独立秒级编号，不覆盖首轮证据。不把首轮记为硬件性能失败或已完成性能测试。
