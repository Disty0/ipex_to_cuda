# ipex_to_cuda

**Run PyTorch CUDA code on Intel XPU – zero code changes required.**

No IPEX package needed. No rewriting `.to("cuda")` calls. Just add two lines and your existing CUDA code runs on Intel GPUs.

---

## Why this exists

You have PyTorch code written for NVIDIA GPUs:
```python
model.to("cuda")
torch.cuda.synchronize()
torch.cuda.empty_cache()
```

You want to run it on Intel hardware (Arc, MAX, Battlemage). Instead of hunting down every CUDA reference, let this hijacker handle the redirect automatically.

---

## Quick start

### 1. Clone into your project

```bash
# From your project root
mkdir -p modules
cd modules
git clone https://github.com/Disty0/ipex_to_cuda.git
```

Your structure:
```
your_project/
├── modules/
│   └── ipex_to_cuda/
│       └── src/
│           └── ipex_to_cuda/
│               └── __init__.py
├── main.py
└── ...
```

### 2. Initialize in your entry point

**In the very first script that runs** (before importing any model code):

```python
# main.py
import torch

if torch.xpu.is_available():
    from modules.ipex_to_cuda.src.ipex_to_cuda.__init__ import ipex_init
    ipex_init()
    print("✓ CUDA → XPU hijacking active")

# Now import and run your CUDA code
from your_model import load_model
model = load_model()
model.to("cuda")  # Actually goes to XPU
```

### 3. Run as normal

```bash
python main.py
```

Your existing `.to("cuda")`, `.cuda()`, and `torch.cuda.*` calls now work on Intel XPU.

---

## What gets hijacked

| Your code | Actually runs |
|-----------|---------------|
| `model.to("cuda")` | `model.to("xpu")` |
| `tensor.cuda()` | `tensor.xpu()` |
| `torch.cuda.is_available()` | `torch.xpu.is_available()` |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` |
| `torch.cuda.empty_cache()` | `torch.xpu.empty_cache()` |
| `torch.cuda.device_count()` | `torch.xpu.device_count()` |

All other tensor operations remain untouched.

---

## Complete example

**Project structure:**
```
my_llm_app/
├── modules/
│   └── ipex_to_cuda/     (cloned repo)
├── models/
│   └── llama_runner.py   (has .to("cuda") calls)
└── run.py
```

**run.py (entry point):**
```python
import torch

# Hijack FIRST
if torch.xpu.is_available():
    from modules.ipex_to_cuda.src.ipex_to_cuda import ipex_init
    ipex_init()

# Then import your CUDA-dependent code
from models.llama_runner import LlamaRunner

# Everything just works
runner = LlamaRunner()
runner.load()  # Contains .to("cuda") inside
output = runner.generate("Hello world")
```

**models/llama_runner.py (unchanged):**
```python
class LlamaRunner:
    def load(self):
        self.model = AutoModel.from_pretrained("meta-llama/...")
        self.model.to("cuda")  # ← Hijacked to XPU
        return self
```

---

## Verification

Add this after `ipex_init()` to confirm hijacking:

```python
if torch.cuda.is_available() and hasattr(torch.cuda, "is_xpu_hijacked"):
    test = torch.randn(3, 3).to("cuda")
    print(f"✓ Tensor on: {test.device}")  # Should show 'xpu:0'
    print("✓ Hijack working")
```

---

## Environment variables (for Intel Arc GPUs)

Tune memory usage for consumer Intel GPUs:

| Variable | Purpose | Default | When to change |
|----------|---------|---------|----------------|
| `IPEX_SDPA_SLICE_TRIGGER_RATE` | Dynamic attention slice trigger (GB) | `1` | Lower for Arc GPUs with 8GB or less |
| `IPEX_ATTENTION_SLICE_RATE` | Slice size for 32-bit GPUs (GB) | `0.5` | Reduce if you see OOM errors |
| `IPEX_FORCE_ATTENTION_SLICE` | Force enable/disable dynamic attention | `0` | Set `1` for Arc/Battlemage, `-1` to disable |

**Pro tip for Intel Arc A770 (16GB):**
```bash
export IPEX_FORCE_ATTENTION_SLICE=1
export IPEX_ATTENTION_SLICE_RATE=0.3
python run.py
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| Hijack not working | Imported model before `ipex_init()` | Move `ipex_init()` to absolute top of entry point |
| CUDA errors persist | Edge function not hijacked | Open GitHub issue with the function name |
| Out of memory | Attention slicing disabled | Set `IPEX_FORCE_ATTENTION_SLICE=1` |


---

## Requirements

- PyTorch with XPU support (`pip install torch --index-url https://download.pytorch.org/whl/xpu`)
- Intel GPU (Arc, MAX, Battlemage, or integrated with XPU support)
- Python 3.8+

---

## Contributing

Found a CUDA function that isn't hijacked? Open an issue with:
- The function name (e.g., `torch.cuda.Stream()`)
- A minimal code example
- Your PyTorch version

---

## License

MIT
