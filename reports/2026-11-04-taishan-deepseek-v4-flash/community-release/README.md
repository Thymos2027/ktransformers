# 社区逐层 MXFP4 专家 GGUF 发布准备

目标是整理可审阅的 ModelScope 社区模型仓库候选包。这里的文件尚未上传，也未创建外部仓库。

公开权重对比与定位见 [社区现有权重核查](community-availability-20261104.md)：公开整模型 MXFP4 GGUF 已存在，本包的价值是对应 KT 路径的逐层 CPU 专家交付与验证，不宣称首发或唯一。

- 候选模型卡与配套文件：[`repository/`](repository/README.md)。
- 服务器候选包目录：`/data/deploy/dsv4-community-release-20261104/repository`。
- 服务器状态：`/data/deploy/dsv4-community-release-20261104/status.json`。
- 全量审计日志：`/data/deploy/dsv4-community-release-20261104/audit.log`。
- 审计范围：43 个 GGUF 的 SHA-256、全 43 层 × 256 专家 × 3 投影的上游 bit-exact 验证，以及独立的打包字节/缩放比较；通过后才进入 `ready_for_review`。

本地草稿不包含约 147.17 GB 的 GGUF。服务器在审计通过后通过硬链接加入已有权重，不额外复制一份；候选包和运行目录共享 inode，打包后不应就地修改。

模型数据保留 DeepSeek MIT 许可证；附带的 KT 转换/验证脚本使用 Apache-2.0，见候选包中的 `LICENSE`、`LICENSE.tools`、`NOTICE`。没有添加额外使用限制。

后续：读取完整审计结果 → 同步模型卡和证据 → 用户审阅并确定 ModelScope namespace/repository ID → 发布。当前没有创建或上传外部仓库，不读取或输出账号 token。
