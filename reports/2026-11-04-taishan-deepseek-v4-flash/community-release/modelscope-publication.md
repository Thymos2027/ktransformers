# ModelScope 后台发布任务

2026-11-04 18:46 最终结果：ModelScope 发布完成，状态 `published_and_verified`。43 个 GGUF 共 147,169,763,520 字节；全部 59 个文件的远端大小及 SHA-256 匹配。

- [ModelScope 仓库](https://modelscope.cn/models/kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts)
- [发布校验记录](../artifacts/modelscope-publication-result.json)
- [上游文档 PR #2257](https://github.com/kvcache-ai/ktransformers/pull/2257)，已创建，尚未合并。

以下早期进度快照保留为历史记录，以此最终结果为准。

授权：2026-11-04 用户明确同意分享到 ModelScope，并在项目 `.env` 配置写权限令牌。`.env` 已被 Git 忽略；本任务未修改该文件或 `.gitignore`。

账号认证成功：`kmon03f`。计划仓库：`kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts`；计划 URL 为 `https://modelscope.cn/models/kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts`。启动任务前确认该仓库尚不存在；创建前会再次检查，拒绝覆盖非本任务创建的已有仓库。

2026-11-04 10:12 快照：完整审计已通过 10/43 层；发布任务为 `waiting_for_full_audit`，尚未创建/上传仓库。此文件不是实时状态。

2026-11-04 11:15 更新：43 层、33,024 个专家投影全部通过，独立 SHA-256 重读及审计后生成验证通过。上传任务已创建仓库，进入 `uploading`；11:19 检查时容器持续运行且已累计产生约 900 MB 发送流量，尚无已验收发布结果。SDK 在大文件传完后计入完成字节，网络发送量不等于已提交/远端校验完成量。

审计证据已同步至项目：[完整结果](repository/validation/full-audit.json)、[审计后生成](repository/validation/post-audit-runtime.json)、[43 文件 SHA-256](repository/SHA256SUMS)。

## 自动执行顺序

1. 在独立 SDK 目录安装官方 `modelscope-hub==0.6.1`，核对认证账号。
2. 等待审计 `ready_for_review`。检查全 43 层、33,024 个专家投影、编码/缩放与解码逐元素精确比较、独立 SHA-256 复核、审计后生成全部通过。
3. 创建公开 MIT 社区转换仓库，上传候选包；不上传 W8A8、源 checkpoint、任务凭据或日志目录。
4. SDK 使用上传缓存、两文件并发、内置恢复，并额外进行有界重试；不删除远端文件。
5. 核对远端 43 个 GGUF 的大小及 SHA-256；通过后更新发布元数据，再验证所有已上传文件。
6. 仅在完整验证后写入 `published_and_verified` 与 `publication-result.json`。中途失败保留状态、上传缓存与错误类型，不能当作发布成功。

完整审计失败、账号不符或不完整的审计记录阻断发布，这三条已通过离线假 API 测试；假令牌未出现在日志，凭据暂存文件读取后删除也通过测试。

## 后台位置

- 审计容器：`dsv4-community-audit-20261104`。
- 发布容器：`dsv4-modelscope-publish-20261104`，2 CPU 配额、4 GiB 内存上限。
- 发布任务根目录：`/data/deploy/dsv4-modelscope-publish-20261104`。
- 审计状态：`/data/deploy/dsv4-community-release-20261104/status.json`。
- 发布状态：`/data/deploy/dsv4-modelscope-publish-20261104/status.json`。
- 上传日志/缓存：`publish.log`、`upload-tracker.json`，位于发布任务根目录。
- 完成记录：同目录 `publication-result.json`。

```powershell
ssh taishan 'cat /data/deploy/dsv4-community-release-20261104/status.json'
ssh taishan 'sudo -n cat /data/deploy/dsv4-modelscope-publish-20261104/status.json'
ssh taishan 'sudo -n docker logs --tail 20 dsv4-modelscope-publish-20261104'
```

发布任务目录权限为 0700；部分状态和日志文件仅 root 可读，使用上述 sudo 读取。上传网络使用本机 LAN 代理 `192.168.137.1:7890`，需要维持该代理及网络可用。

## 凭据处理

令牌从本地 `.env` 在进程内读取，经 SSH 标准输入传到 taishan，未放入命令行、报告、Git 或 Docker 环境配置。临时文件以 0600 暂存；发布进程读取后已删除，令牌仅保留在进程内存，用于这次上传。未调用全局 SDK login 保存登录信息。

容器退出后若需重启上传，需要重新从用户仍保留的 `.env` 安全注入凭据；不能直接重启并假定凭据仍存在。不要将发布任务目录整体打包为公开仓库。
