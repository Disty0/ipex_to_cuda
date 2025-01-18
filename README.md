# ipex_to_cuda
Adapt IPEX to CUDA


Converts torch.cuda or model.to("cuda") to torch.xpu or model.to("xpu") automatically.  

Add IPEX support without extra code changes:
```python
import torch
try:
    import intel_extension_for_pytorch as ipex
except Exception:
    pass

if torch.xpu.is_available():
    from ipex_to_cuda import ipex_init
    ipex_active, message = ipex_init()
    print(f"IPEX Active: {ipex_active} Message: {message}")


if torch.cuda.is_available():
    if hasattr(torch.cuda, "is_xpu_hijacked") and torch.cuda.is_xpu_hijacked:
        print("IPEX to CUDA is working!")
    torch_model.to("cuda")
```



### Environment Variables

- `IPEX_SDPA_SLICE_TRIGGER_RATE`: Specify when dynamic attention slicing for Scaled Dot Product Attention should get triggered for Intel ARC. This environment variable allows you to set the trigger rate in gigabytes (GB). The default is `4`.

- `IPEX_ATTENTION_SLICE_RATE`: Specify the dynamic attention slicing rate for 32 bit GPUs. This environment variable allows you to set the slicing rate in gigabytes (GB). The default is `4`.

- `IPEX_FORCE_ATTENTION_SLICE`: Specify to enable or disable Dynamic Attention. Useful for saving memory with Intel GPU MAX and Battlemage series. The default is `0`.
  - `1` will force enable dynamic attention slicing even if the GPU supports 64 bit.
  - `-1` will force disable dynamic attention slicing even if the GPU doesn't support 64 bit.
  - `0` will automatically enable or disable dynamic attention based on the GPU.
