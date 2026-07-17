# Profiling GPT-2 Training: A Practical Guide

This guide helps you identify which part of the training pipeline is bottlenecking — whether that's compute, memory bandwidth, data loading, or communication.

---

## 1. Quick Sanity Checks

### GPU utilization (`nvidia-smi`)

```bash
watch -n 1 nvidia-smi
```

Look at the **GPU-Util** column during training:

| Utilization | Interpretation |
|:---:|---|
| 90–100 % | GPU is saturated — you're compute- or memory-bound. Good. |
| 50–90 % | Partially utilized. Could be data loading, gradient sync, or small matrices. |
| < 50 % | The GPU is starved. Likely data loading, DDP sync, or a CPU bottleneck. |

### VRAM check

```python
# Add to your training loop
print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

If memory is low (< 50 % of VRAM) and GPU util is low, the model is too small to saturate the GPU. Increase batch size, context length, or model size before optimizing kernels.

### Step timing breakdown

Insert these timers into the training loop to see where time is spent:

```python
import time

for step in range(max_iters):
    t_start = time.time()

    # --- forward ---
    torch.cuda.synchronize()
    t_fwd_start = time.time()
    logits, loss = model(x, y)
    torch.cuda.synchronize()
    fwd_time = time.time() - t_fwd_start

    # --- backward ---
    torch.cuda.synchronize()
    t_bwd_start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.time() - t_bwd_start

    # --- optimizer step ---
    torch.cuda.synchronize()
    t_opt_start = time.time()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    opt_time = time.time() - t_opt_start

    # --- data loading (overlaps but worth measuring) ---
    torch.cuda.synchronize()
    t_load_start = time.time()
    x, y = get_batch("train")
    torch.cuda.synchronize()
    load_time = time.time() - t_load_start

    total_time = time.time() - t_start
    print(f"step={step} fwd={fwd_time*1000:.1f}ms bwd={bwd_time*1000:.1f}ms opt={opt_time*1000:.1f}ms load={load_time*1000:.1f}ms total={total_time*1000:.1f}ms")
```

| Profile | Interpretation |
|---|---|
| `fwd` dominates | Forward pass kernel launch overhead or inefficient kernels |
| `bwd` is ~2–3x `fwd` | Normal — backward pass is typically 2–3x the forward pass |
| `opt` is significant | Optimizer step is consuming time. Use fused AdamW. |
| `load` is significant | Data loading / CPU-GPU transfer is bottlenecking. |
| Any slot idle while others run | GPU is waiting for CPU work — CPU-bound. |

---

## 2. Compute vs. Memory Bound — The MFU Test

Your [model.py](model.py) has `estimate_mfu()` (line 289). Check **MFU** (Model Flops Utilization) from logs:

```
iter 100: loss 2.5000, time 52.34ms, mfu 45.23%
```

| MFU | Diagnosis |
|---|---|
| > 50 % | Efficient. You're using the GPU well. |
| 20–50 % | Some inefficiency. Likely small matrices or kernel launch overhead. |
| 5–20 % | Model or batch is too small for the GPU. |
| < 5 % | Something is very wrong — tiny model, CPU fallback, etc. |

**Key insight**: For GPT-2 124M on a single GPU, attention (~80% of all FLOPs) is the compute-heavy operation. LayerNorm + residual connections + embedding lookup are **memory-bandwidth bound**. Small batches / short contexts produce low MFU because the GPU can't saturate its compute units.

**Improve MFU** by increasing: batch size, `block_size`, or model size.

---

## 3. torch.profiler — Kernel-Level Profile

```python
from torch.profiler import profile, ProfilerActivity

profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)
profiler.start()

# Run 10–15 steps (skip first 1–2 for warmup)
for step in range(15):
    ...
    if 2 <= step <= 15:
        profiler.step()
    ...

profiler.stop()
print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30))
profiler.export_chrome_trace("profile_trace.json")
```

Open `profile_trace.json` in Chrome at `chrome://tracing` or https://ui.perfetto.dev.

**What to look for:**

| Kernel | Significance |
|---|---|
| `cublas::gemm` | Matmul kernels. Should dominate (~60–70 % of CUDA time). |
| `cublasGemm` with small M/K/N | Small matmuls that launch poorly. Increase batch/context. |
| `LayerNorm` / `native_layer_norm` | If > 10 %, normalization is too frequent or batch is too small. |
| `flash_attn_varlen_fwd` | FlashAttention kernel. Should be the dominant attention compute. |
| `triton_kernel_launch` | Triton kernel execution time. Check if it's efficient. |
| `cudaMemcpy` / `memcpy` | Host-device transfer. If high, data loading is bottlenecking. |
| `nccl::` | DDP all-reduce. If > 20 %, communication is the bottleneck. |

---

## 4. nsys — Timeline View

```bash
nsys profile --trace=cuda,nvtx,osrt,ntf -o profile ./train.py --batch_size=64
nsys ui profile.qdstrace &
```

**In the timeline, look for:**

- **Gaps between kernel launches** — GPU is idle, waiting for CPU. Python loop or data loading is too slow.
- **`cudaMemcpy` blocks** — Host-device transfer. Use `pin_memory()` + `non_blocking=True` to overlap.
- **`nccl` kernels between compute** — Communication fragmenting compute. Use larger micro-batches.
- **Kernel overlap** — Are forward and backward overlapping well? Long stalls between layers suggest kernel fusion could help.

---

## 5. Nsight Compute (ncu) — GPU Microarchitecture Analysis

```bash
ncu --set full -o attention_profile python -c "
import torch
from model import GPTConfig, GPT
model = GPT(GPTConfig())
model.cuda()
x = torch.randint(0, 50304, (4, 1024), device='cuda')
y = torch.randint(0, 50304, (4, 1024), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    _, loss = model(x, y)
    loss.backward()
"
```

**Key metrics:**

| Metric | Good | Bad |
|---|---|---|
| L2 cache hit rate | > 70 % | < 50 % |
| Achieved occupancy | > 70 % | < 50 % |
| Memory throughput | Near peak for attention | Far below peak (compute-bound on small mats) |
| HBM write volume | Low (fused kernels) | High (intermediate activations spilled) |
| Achieved occupancy | Low (< 50 %) — register/memory bound |
| HBM write volume | Low (fused kernels) | High (intermediate activations spilled) |

---

## 6. Per-Component Bottleneck Matrix

Where time is typically spent in a GPT-2 training step:

```
                    Forward   Backward    Optimizer   Data Load   All-Reduce
Matmul (c_attn)      ████      █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Matmul (c_proj)      ████      ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LayerNorm            ██        ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Embedding lookup     ██        ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
AdamW step           ░░        ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
Gradient clipping    ░░        █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
All-reduce (DDP)     ░░░░      ░░░░░░░░░░░░░░░░░░░░░░██████████████████
```

| Dominant component | Action |
|---|---|
| Matmul (~80 %) | Use FlashAttention, larger batch, `torch.compile` |
| AdamW (10–30 %) | Use fused AdamW (`fused=True`), DeepSpeed, or lower precision |
| LayerNorm (5–15 %) | Fuse LayerNorm with the subsequent matmul |
| Data load (< 5 %) | Already fine with `pin_memory` + `non_blocking` |
| All-reduce (DDP) | Use larger micro-batches, gradient compression, or pipeline/tensor parallelism |

---

## 7. Practical Checklist — Run in Order

1. **Check `nvidia-smi` util during training** — Is GPU > 80 %?
2. **Check MFU from logs** — Is it > 30 %? If not, increase batch or context.
3. **Run step timing breakdown** — Which sub-step is slowest?
4. **Run `torch.profiler`** — What's the top CUDA kernel by time?
5. **Check L2 hit rate with `ncu`** — Is memory access efficient?
6. **Check for nccl communication overhead** — Is all-reduce dominating?
7. **Profile data loading** — Are you waiting on the CPU?

---

## 8. Expected Baseline Numbers (A100 80GB)

| Config | Tokens/sec | MFU | Step time |
|---|---|---|---|
| 124M, B=64, T=1024, `torch.compile` | ~30K–40K | 40–55 % | ~30–50 ms |
| 124M, B=64, T=1024, no `torch.compile` | ~15K–25K | 20–35 % | ~50–80 ms |
| 124M, B=32, T=512, `torch.compile` | ~10K–15K | 25–40 % | ~50–80 ms |
| 124M, B=8, T=128, `torch.compile` | ~2K–4K | 5–15 % | ~200–500 ms |

If your numbers are far below these, profiling will reveal the specific bottleneck.