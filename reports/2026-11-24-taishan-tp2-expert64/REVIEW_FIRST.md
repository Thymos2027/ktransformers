# 首轮启动失败：版本检查中的导入顺序

运行：`expert64-20261124-first`。本机终态于 2026-11-24 09:09:48（北京时间）记录为 `failed`，`results_downloaded=true`。远端时钟较本机约快 30 秒，不据此排列不同机器上的事件先后。

首轮在 `hccl_probe` 阶段退出，但实际 HCCL 程序未执行。`logs/hccl.log` 显示 SGLang 从隔离源码成功导入后，新增的版本检查直接 `import kt_kernel_ext`，产生 `ModuleNotFoundError`，shell 因非零退出码终止。

固定 KT 源码 `kt-kernel/python/__init__.py` 先通过 `_initialize_cpu()` 加载本机后端，再将其注册为 `sys.modules['kt_kernel_ext']`；正确方式是先 `import kt_kernel`，再通过 `from kt_kernel import kt_kernel_ext` 获取实际扩展。这是本轮新增检查命令的错误，不能归因为依赖缺失、HCCL 通信失败或 K64 不受支持。

失败前未加载模型，没有执行正确性或性能请求。远端 `cleanup.json` 显示实验容器已停止、原服务仍停止、18082 端口释放、两张 NPU 均无运行进程。用户反馈后独立核查亦为空闲，两卡健康 `OK`、62°C/58°C。首轮 12 次有效风扇采样无异常。

修复将版本检查移入 `environment_probe.py`，通过正常初始化取得扩展并输出实际库 SHA-256；检查 factory `__new__` 的参数及 LLAMAFILE 后端继承的必要方法。HCCL 阶段独立执行通信探针，模型阶段执行 KT 环境检查。保留首轮所有证据，以新的 run ID 重试，复用相同镜像、权重、参数、TP2 补丁及热保护，不安装或升级依赖。
