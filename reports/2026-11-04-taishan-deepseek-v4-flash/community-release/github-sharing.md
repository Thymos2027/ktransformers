# 上传完成后的 KT 上游分享

2026-11-04 18:46 最终结果：ModelScope 发布完成，状态 `published_and_verified`。43 个 GGUF 共 147,169,763,520 字节；全部 59 个文件的远端大小及 SHA-256 匹配。

- [ModelScope 仓库](https://modelscope.cn/models/kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts)
- [发布校验记录](../artifacts/modelscope-publication-result.json)
- [上游文档 PR #2257](https://github.com/kvcache-ai/ktransformers/pull/2257)，已创建，尚未合并。

以下早期进度快照保留为历史记录，以此最终结果为准。

用户于 2026-11-04 授权：ModelScope 上传完成后，可在 KTransformers 上游 Issue 或 PR 分享。

选择文档 PR：上游 `.github/ISSUE_TEMPLATE/config.yml` 禁用空白 Issue，只提供 bug/feature 模板。资源下载链接更适合补入现有 Ascend 教程，而非新增无代码改动的 PR 或占用 bug Issue。此前检索未发现指向本候选包的重复 Issue/PR。

目标仓库：`kvcache-ai/ktransformers`。已确认连接器与本机 GitHub CLI 都以 `Thymos2027` 登录，个人 fork `Thymos2027/ktransformers` 有写权限。

计划 PR：`docs: link community MXFP4 CPU expert weights for Ascend`。

- 修改 `doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md`。
- 在权重工件表后增加一个社区预转换专家文件下载段落。
- 明确第三方社区维护、仅含 CPU 专家、仍需 W8A8、固定版本与哈希校验。
- 不向上游提交大体积 GGUF 或其他用户本地文件。
- 不指定审阅人、自动合并或发送其他通知。

可审阅材料：[拟提交的文档改动](github-pr-proposed-change.diff)、[PR 正文](github-pr-body.md)。正文中的完整审计和上传验证陈述，只有在下列条件成立后才允许发出。

## 后台触发条件

1. taishan 发布任务为 `published_and_verified`，完整结果确认 43 个 GGUF 均在 ModelScope 端匹配大小/SHA-256。
2. 从本机以无令牌会话重新读取公开 ModelScope 文件列表，核对已发布的全部文件，再读取公开的全量审计报告。
3. 检查 GitHub CLI 登录仍为 `Thymos2027`。
4. 查重：已有本分支 PR 则记录链接；教程已经含该下载链接则不重复创建。
5. 从届时的上游 main 创建个人 fork 文档分支，保留原文件换行格式，只修改一个段落，然后创建文档 PR。

任一审计/发布失败、账号变化、上游文档结构变化或校验失败，均停止分享并记录原因。

## 运行位置

本机后台目录：`D:\Models\DeepSeek\_github-share-20261104`。

- `status.json`：等待/校验/PR 创建结果；成功为 `pr_opened`，包含 `pr_url`。
- `share.log`、`stderr.log`：日志。
- `watcher.pid`：本机进程号。
- `proposed-change.diff`、`pr-body.md`：拟提交内容。

该任务使用本机已登录的 `gh` 执行延后触发的提交，不读取、复制或向 taishan 传输 GitHub 令牌。连接器已用于账号、上游仓库与模板核对；脱离聊天会话后的条件执行由 GitHub CLI 完成。

后台每分钟检查一次发布结果，最多等待七天；网络连续不可达 60 次或发布失败会停止。它脱离当前聊天会话运行，但本机需要保持开机和网络可用；未配置 Windows 重启后自动恢复。

当前仅已准备文档并启动等待任务，**尚未创建 GitHub PR**。
