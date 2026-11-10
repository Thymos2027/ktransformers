# taishan 重启后恢复核查

核查与操作：2026-11-10（北京时间）。用户已授权恢复 Docker、排查并尝试恢复 Ascend 驱动服务，以及在硬件可用后启动原容器。SSH 使用 `taishan` 和 `taishan-ibmc` 配置别名；没有读取密钥内容。

## 事实

- 主机最近一次启动约为 2026-11-09 20:49。Docker 服务和 socket 的 mask 链接生成于 19:01，即重启之前；原容器 `dsv4-kt-20261104` 已退出，重启策略为 `no`。
- `host_sys_init.service` 在本次启动时等待 5 分钟后超时。`/dev/davinci_manager` 缺失、Ascend 模块未加载，`npu-smi info` 报驱动初始化失败。
- 上次正常启动的内核日志枚举到两张 910B，PCI 地址 `0000:01:00.0` 和 `0000:81:00.0`，设备 ID `19e5:d802`。当前启动的 `lspci` 与 PCI 树在这两处均无终端设备；对应根端口仍存在，链路状态显示 2.5 GT/s ×1，而以前两卡为 16 GT/s ×16。手动 PCI 总线重扫后，两卡仍未出现。
- iBMC 电源状态为 On，健康事件报告一项 PCIe 卡 7 的 BBU 故障；现有信息不能把该 BBU 事件归因于两张 NPU 消失。

## 已执行的恢复操作

1. `systemctl unmask docker.service docker.socket`，`systemctl daemon-reload`，`systemctl start docker.service`。Docker 现为 `active`，原容器仍保持停止。
2. 对 `/sys/bus/pci/rescan` 写入 `1`，重扫 PCIe。两张 910B 仍未枚举。

没有启动原容器，也没有再次启动 `host_sys_init.service`：当前系统层面看不到两张 NPU，单独重试驱动初始化预计只会重复超时。没有执行系统重启、断电、iBMC 电源操作或硬件热插拔。

## 当前阻碍与后续验收

目前不能恢复模型服务，也不能重跑 32 线程实验。下一步需要确认两张 NPU 的供电、PCIe 链路和硬件状态；可能需要受控的冷断电再上电或现场检查，这会中断整台服务器，应单独授权。只有 `lspci` 重新枚举两张 `19e5:d802`、`npu-smi info` 识别两卡，才启动原容器并做 `/health` 和短请求验收，然后在新的运行目录重跑干预实验。

## 2026-11-10 冷断电恢复任务

用户随后明确授权整机冷断电再上电。已启动本机隐藏后台流程 `recover.py`，PID 16456；运行状态写入 `runs/first/status.json`，命令记录在 `runs/first/commands.log`，标准输出与错误分别在 `recover.stdout.log`、`recover.stderr.log`。流程先正常关机，确认 Off 后等待 30 秒再开机；只有两张 NPU 重新枚举、`npu-smi` 正常时才启动原容器，并检查 `/health` 和短请求。任何门槛失败即停止后续步骤；若发现仍处于 Off，会尝试恢复上电。

18:26:19 iBMC 接受了正常关机命令；短暂检查时仍显示 On，流程处于 `powering_off`。此为进行中状态，不表示冷断电或服务恢复已完成。完成结果以后续状态文件和实时核查为准。

## 冷启动后复核（18:42）

后台状态已走过 `powered_off`（18:29:26）、`powering_on`（18:29:56）和 `waiting_for_host`；主机本次启动时间为 18:32:51，iBMC 电源为 On。任务仍处于 `waiting_for_npu`，最多等待 15 分钟，尚未进入容器启动阶段。

本次启动的 `lspci` 仍找不到两张 `19e5:d802`（原 PCI 地址 `01:00.0`、`81:00.0` 均空），两个对应根端口均只报告 2.5 GT/s ×1，未建立原有的 16 GT/s ×16 链路。`host_sys_init.service` 再次超时，`/dev/davinci_manager` 不存在，`npu-smi` 初始化失败。BIOS 版本与先前相同，均为 6.70（2024-04-03）。Docker 活动正常，原容器仍停止。iBMC 唯一当前健康事件为 PCIe 卡 7 的 BBU 故障；传感器将其标为 PCIe RAID7，不能据此认定是 NPU 故障。

两卡在 PCIe 枚举阶段就缺失，故 Ascend 驱动失败更可能是下游表现；固件、卡供电、PCIe 连接、主板/扩展板配置或硬件故障仍需要分别核查。当前证据不支持直接升级或重装驱动、刷新固件。待后台任务给出最终状态后，应检查 iBMC 的 NPU/PCIe 插槽清单和历史事件，必要时现场核对卡与供电。

## 后续恢复（19:21）

冷启动任务于 18:48:27 达到 15 分钟的 NPU 等待期限并以 `PCIe count=0` 失败，故未启动容器。之后设备状态发生变化；19:21:54 再次核查时，`01:00.0` 与 `81:00.0` 的两张 `19e5:d802` 均已重新枚举，`npu-smi` 显示两张 910B4-1 均为 `OK`、无模型进程，Ascend 初始化服务和 Docker 均为 `active`。目前不能从这些快照确定设备如何恢复，之前的硬件原因仍未查明。

已按用户要求启动原容器 `dsv4-kt-20261104`，它会重新执行固定版本部署 pipeline。原服务健康与短请求验收，以及其后的 32 线程干预实验，由[实验后台控制器](../2026-11-08-taishan-dual-instance-thread-budget/retry_controller.py)顺序执行；此处不将容器启动视为服务恢复验收完成。

## 19:29–19:32 温度与风扇异常

原容器重新执行部署 pipeline 并开始加载模型后，`npu-smi` 只列出物理 NPU 5；19:29 它在无模型进程、AICore 0% 时为 103°C、127 W、Health OK。19:30:23 原容器停止后，NPU 5 仍升至 111°C、140 W、Health Alarm。NPU 2 当时仍在 `lspci`，但已不在 `npu-smi` 列表。19:30:59 起 `npu-smi` 报 `dcmi module initialize failed`，不能继续读取卡温。内核日志在 19:29 报 NPU 2 心跳丢失（Critical）和 DMA 错误，19:30 报 NPU 5 设备中断无响应。

iBMC 风扇控制当时为 `auto`，四个机箱风扇约 3000–3075 RPM，状态均正常；进风 23°C、出风 39°C，iBMC 传感器列表未看到 NPU 温度项目。自动模式没有随上述 NPU 读数升速。可能是 BMC 未将 NPU 温度作为风扇调速输入，尚无风扇控制策略证据可确认具体原因。

用户明确授权临时提高风扇转速后，19:31 将 iBMC 切为手动模式并设全体风扇 100%；约 15 秒后测得四个风扇约 14,925–15,450 RPM，设置确实生效。手动模式设有 1800 秒超时，到期自动回到自动调速。原容器已停止，本机实验后台控制器已终止并将状态标为 `cancelled_thermal_alarm`；`runs/second/` 未创建，32 线程实验未开始。此时硬件和驱动状态不满足重跑门槛，不应启动模型服务。

用户随后授权关闭整机电源。19:34:01 iBMC 接受正常关机命令；隐藏后台任务 `shutdown_for_thermal.py` 于 19:34:26 在 `runs/thermal-shutdown/status.json` 记录 `complete`、`power: Off`。命令日志为 `runs/thermal-shutdown/commands.log`，标准输出/错误分别为 `shutdown.stdout.log`、`shutdown.stderr.log`。任务只执行关机，没有自动重新上电。当前 NPU 故障原因和自动风扇未升速的具体机制尚未查明，重新上电及模型实验须在硬件/散热排查后另行决定。
