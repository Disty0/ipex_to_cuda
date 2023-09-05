# ipex_to_cuda
Adapt IPEX to CUDA

Add IPEX support without affecting CUDA users:
```
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from ipex_to_cuda import ipex_init
        ipex_init()
except Exception:
    pass
```
