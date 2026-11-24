# 首轮 EP2：初始化时的 CPU/NPU 张量设备冲突

运行 `ep2-expert64-20261124-first`，2026-11-24。环境检查、两rank HCCL均通过，随后在创建第一个MoE层的wrapper时失败，没有完成模型加载、驻留审计或生成请求。

直接错误是新增代码 `torch.equal(self.gpu_experts_mask.cpu(), torch.tensor(cpu_skip))`。模型加载在默认设备为NPU的上下文内执行，未指定device的新张量被建在对应NPU，而比较左侧明确在CPU，两个rank均报设备不一致。此前的纯Python布局测试及语法编译不能覆盖这个PyTorch默认设备行为。

本轮 `cleanup_verified=true`：实验容器停止、原服务保持停止、18082释放、两卡无计算进程；只读回查健康OK、约63°C/58°C，没有温度中止或OOM依据。原始目录、日志及失败状态保留，不覆写为成功。

修复在EP2新增的CPU mask、slot表和CPU identity arange中全部显式指定 `device="cpu"`。放到NPU上的路由表仍由原有加载后步骤执行。新环境阶段增加 `device_probe.py`：在真实镜像PyTorch中，提取实际生成的wrapper构造函数和CPU mapping门禁，模拟非CPU默认设备（meta），先重现未修复构造的失败，再确认两个rank的修复构造均保留CPU表及正确归属。该probe不加载完整模型、不代替NPU模型数值验证；通过后才进入HCCL及模型试验。

新增AST回归检查所有引入的tensor/arange工厂都有明确device。新运行标识为 `ep2-expert64-20261124-device-fix`，模型、专家分布、EP/TP参数、CPU计算集合及现有温控规则保持。
