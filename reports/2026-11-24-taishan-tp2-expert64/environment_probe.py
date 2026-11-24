"""Check the installed KT API through its package initializer before using aliases."""
import hashlib
import inspect
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    import kt_kernel

    # kt_kernel initializes the selected native backend and registers this alias.
    # A direct import of kt_kernel_ext in a fresh process is not supported.
    from kt_kernel import KTMoEWrapper, kt_kernel_ext
    from kt_kernel_ext.moe import MOE, MOEConfig
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper

    # The public entry point is a __new__ factory, returning a backend wrapper.
    for target in (KTMoEWrapper.__new__, LlamafileMoEWrapper.__init__):
        parameters = inspect.signature(target).parameters
        for name in ('gpu_experts_mask', 'swiglu_limit'):
            if name not in parameters:
                raise RuntimeError(f'{target.__qualname__} is missing {name}')
    required = ('copy_inputs_to_cpu_buffers', 'run_pinned_forward_sync',
                'copy_forward_output_to_device')
    for name in required:
        if not callable(getattr(LlamafileMoEWrapper, name, None)):
            raise RuntimeError(f'KT wrapper is missing {name}')
    extension = Path(kt_kernel_ext.__file__)
    result = dict(passed=True, torch=torch.__version__, torch_npu=torch_npu.__version__,
                  kt_package=kt_kernel.__file__, kt_extension=str(extension),
                  extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                  required_methods=list(required), native_moe=MOE.__name__,
                  native_config=MOEConfig.__name__)
    print('KT environment: ' + json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
