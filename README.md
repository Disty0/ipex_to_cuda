# ipex_to_cuda
Adapt IPEX to CUDA


Converts torch.cuda or model.to("cuda") to torch.xpu or model.to("xpu") automatically.  

Add IPEX support without extra code changes:
```python
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from ipex_to_cuda import ipex_init
        ipex_active, message = ipex_init()
        print(f"IPEX Active: {ipex_active} Message: {message}")
except Exception:
    pass

if torch.cuda.is_available():
    if hasattr(torch.cuda, "is_xpu_hijacked") and torch.cuda.is_xpu_hijacked:
        print("IPEX to CUDA is working!")
    torch_model.to("cuda")
```



### Environment Variables

- `IPEX_SDPA_SLICE_TRIGGER_RATE`: Specify when dynamic attention slicing for Scaled Dot Product Attention should get triggered for Intel ARC. This environment variable allows you to set the trigger rate in gigabytes (GB). The default is `4` GB.

- `IPEX_ATTENTION_SLICE_RATE`: Specify the dynamic attention slicing rate for Intel ARC. This environment variable allows you to set the slicing rate in gigabytes (GB). The default is `4` GB.

- `IPEX_FORCE_ATTENTION_SLICE`: Force use dynamic attention slicing even if the GPU supports 64 bit. Useful with Intel Data Center GPU MAX series.
