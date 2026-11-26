"""Compile/load the exact AscendC converter inside the detached probe container."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    directory = Path(os.environ['KT_MXFP4_OP_DIR'])
    expected = {
        'mxfp4_fused_op.py':'4ce0c5e7a850467f0ff13c2237514523ee3f7750d244c53760a1f4284a6ee388',
        'mxfp4_fused_kernel.cpp':'356d5e8df3822908aa6daac0ec91112cb9cdd8f21a3689d67f3a0c6fe67411a0',
    }
    for name, digest in expected.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('AscendC converter source changed: '+name)
    from gguf import GGUFReader
    sys.path.insert(0, str(directory))
    import mxfp4_fused_op as op
    started = time.monotonic()
    library = op.get_lib()
    result = dict(passed=True, source_sha256=expected, library=library._name,
                  library_sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
                  build_and_load_seconds=time.monotonic()-started, gguf_reader=GGUFReader.__name__,
                  validation='Build and dlopen only; model requests separately verify actual streaming execution')
    Path('/experiment/kernel-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
