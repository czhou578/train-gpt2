# Profiling GPT-2 Training: A Practical Guide

This guide helps you identify which part of the training pipeline is bottlenecking — whether that's compute, memory bandwidth, data loading, or communication.

---

## 0. Quick Decision Tree

Use this to narrow down the bottleneck in under 30 seconds:

```
GPU-Util < 80%?
├── VRAM used < 50%?  → Model/batch too small for GPU → increase B, T, or model size
└── VRAM used > 80%?  → GPU is starved → check data loading (pin_memory, prefetch)
GPU-Util > 80%?
├── MFU > 40%?         → GPU well utilized — nothing to optimize here
├── MFU 20–40%?        → Small matmuls or kernel-launch overhead → torch.compile, larger B/T
└── MFU < 20%?         → Something is very wrong → run torch.profiler to find the culprit
```

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

# Create reusable CUDA events for async measurement
load_start = torch.cuda.Event(enable_timing=True)
load_end   = torch.cuda.Event(enable_timing=True)

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

    # --- data loading (overlaps with GPU — use CUDA events) ---
    load_start.record()
    x, y = get_batch("train")
    load_end.record()
    load_end.synchronize()
    load_time = load_start.elapsed_time(load_end) / 1000  # ms → s

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

Your [model.py](../model.py) has `estimate_mfu()` (line 289). Check **MFU** (Model Flops Utilization) from logs:

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
from torch.profiler import profile, ProfilerActivity, schedule

# profile_memory and with_stack are deprecated in PyTorch 2.x — omit them
profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    schedule=schedule(wait=1, warmup=1, active=5, repeat=1),  # skip warmup, capture 5 steps
)
profiler.start()

# Run training loop — profiler collects only the scheduled steps
for step in range(max_iters):
    train_step()  # your normal forward/backward/opt
    profiler.step()

profiler.stop()

# Print the top 30 CUDA kernels by total time
print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Export to Chrome trace for visual inspection
profiler.export_chrome_trace("profile_trace.json")
```

Open `profile_trace.json` in Chrome at `chrome://tracing` or https://ui.perfetto.dev.

> **Tip**: You can also export to TensorBoard with `profiler.distributed_tensorboard_trace()` or use `torch.profiler.tensorboard_trace_handler()` for multi-step profiles that aggregate across steps.

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

### NVTX markers for nsys/ncu readability

Wrap important sections with `torch.profiler.record_function()` to label them in the nsys timeline:

```python
with torch.profiler.record_function("forward"):
    logits, loss = model(x, y)
with torch.profiler.record_function("backward"):
    loss.backward()
```

These appear as named spans in `nsys` and `ncu` timelines, making it obvious where time is spent without guessing from kernel names.

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

---

## 6. Per-Component Bottleneck Matrix

Where time is typically spent in a GPT-2 training step (approximate % of total step time for a single-GPU 124M config with B=64, T=1024):

| Component | Forward | Backward | Optimizer | Data Load | All-Reduce | Total % |
|---|---|---|---|---|---|---|
| Matmul (c_attn) | ~3 % | ~6 % | — | — | — | **~9 %** |
| Matmul (c_proj) | ~3 % | ~5 % | — | — | — | **~8 %** |
| FlashAttention | ~20 % | ~40 % | — | — | — | **~60 %** |
| LayerNorm | ~1 % | ~3 % | — | — | — | **~4 %** |
| Embedding lookup | ~1 % | ~2 % | — | — | — | **~3 %** |
| AdamW step | — | — | ~10–15 % | — | — | **~10–15 %** |
| Gradient clipping | — | — | ~1 % | — | — | **~1 %** |
| Data load | — | — | — | ~1–2 % | — | **~1–2 %** |
| All-reduce (DDP) | — | — | — | — | ~5–20 % | **~5–20 %** |

> Percentages are rough — measure yours. FlashAttention dominates because it handles the O(n²) attention mechanism.

| Dominant component | Action |
|---|---|
| FlashAttention (~60 %) | Already using it — good. If still slow, check seq_len or increase batch. |
| AdamW (10–15 %) | Use fused AdamW (`fused=True`), DeepSpeed, or lower precision. |
| Matmul (15–20 %) | Use `torch.compile` to fuse matmul chains, increase batch size. |
| LayerNorm (3–5 %) | Fuse LayerNorm with subsequent matmul via `torch.compile`. |
| Data load (< 2 %) | Already fine with `pin_memory` + `non_blocking`. |
| All-reduce (5–20 %) | Use larger micro-batches, gradient compression, or pipeline/tensor parallelism. |

---

## 7. torch.compile Profiling Tips

When you enable `torch.compile` (as the baselines table shows it roughly doubles throughput), the profiling picture changes. Here's what to watch for:

### How compiled traces look different

| Before compile | After compile | What it means |
|---|---|---|
| `cublas::gemm` dominates | Triton kernels dominate | Compilation fuses ops into Triton kernels. Seeing mostly Triton = good. |
| Many tiny kernel launches | Fewer, larger kernels | Fusion is working — less launch overhead. |
| No compilation overhead | `torch.compile` recompilation traces | First ~10–20 steps show compilation. If it persists, you hit a cache miss (dynamic shapes, `@torch.jit` trace issues). |

### Spotting recompilation stalls

In `nsys` or Chrome trace, look for long gaps (>1ms) with no GPU activity during the first steps. These are TorchInductor recompiling. If they reappear later, your model has dynamic control flow (e.g., shape changes) that invalidates the compiled cache.

```bash
# Set this to log compile decisions and cache misses
TORCH_COMPILE_DEBUG=1 python train.py
```

Key debug variables:
- `TORCH_COMPILE_DEBUG=1` — logs every compilation event
- `TORCHDYNAMO_VERBOSE=1` — verbose dynamo logs
- `TORCH_INDUCTOR_BREAKPOINT=1` — break on graph breaks (a graph break = performance cliff)

### What to look for in profiler output

| Signature | Problem | Fix |
|---|---|---|
| Many `torch::jit::run_cuda_graphs` calls | Compilation overhead on each step | Increase batch size or use `torch.compile(mode="reduce-overhead")` |
| `triton_kernel_launch` with low occupancy | Triton kernel is memory-bound | Check block sizes; increase `triton.max_tiling` or use `torch.compile(mode="max-autotune")` |
| Graph breaks in `TORCHDYNAMO_VERBOSE` | Model has Python control flow that dynamo can't trace | Replace `if`/`for` with `torch.where` or static shapes |
| No Triton kernels at all | Compilation failed — falling back to eager | Check `TORCH_COMPILE_DEBUG` for error messages |

### Profiling a compiled model with `ncu`

```bash
# Profile a compiled forward/backward — Triton kernels are named by their operation
ncu --set full --kernel-name regex:"triton.*attention.*" python train.py
```

This filters to just the attention-related Triton kernels, making it easy to see occupancy and memory metrics for the most expensive part of the model.

---

## 8. Memory Profiling

Time isn't the only bottleneck — memory fragmentation and OOMs can kill training.

### Quick VRAM breakdown

```python
# Add to training loop
print(torch.cuda.memory_summary(verbosity=1, abbreviate=True))
```

This prints a per-operator allocation table showing which operations use the most memory.

### Memory snapshot (for OOM debugging)

```python
torch.cuda.mem_snapshot()  # returns a snapshot object
```

Or trigger one automatically on OOM:

```python
import os
os.environ['TORCH_CUDA_MEMORY_SNAPSHOT_ENABLE'] = '1'
```

Then open the generated `.pdf` in Chrome `chrome://tracing` to see the exact allocation/deallocation sequence leading up to the OOM.

### Key memory metrics

| Metric | Tool | What to look for |
|---|---|---|
| Peak memory | `torch.cuda.max_memory_allocated()` | Compare to GPU VRAM — if close to limit, reduce batch or use gradient checkpointing |
| Memory fragmentation | `torch.cuda.memory_stats()['reserved_bytes']` vs allocated | Large gap = fragmentation. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Activation memory | `torch.cuda.memory_summary()` | LayerNorm and residual often use more than expected. Fuse with compile. |
| Gradient memory | `torch.cuda.memory_summary()` | Backward doubles activation memory. Gradient checkpointing (`--grad-acck`) trades compute for memory. |

### Reducing memory

```bash
# Enable expandable memory segments (PyTorch 2.x) — reduces fragmentation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py

# Gradient checkpointing — saves ~50% activation memory at ~15% compute cost
# In model.py, wrap transformer blocks with checkpoint_wrapper
```

---

## 9. Practical Checklist — Run in Order

1. **Check `nvidia-smi` util during training** — Is GPU > 80 %?
2. **Check MFU from logs** — Is it > 30 %? If not, increase batch or context.
3. **Run step timing breakdown** — Which sub-step is slowest?
4. **Run `torch.profiler`** — What's the top CUDA kernel by time?
5. **Check L2 hit rate with `ncu`** — Is memory access efficient?
6. **Check for nccl communication overhead** — Is all-reduce dominating?
7. **Profile data loading** — Are you waiting on the CPU?
8. **Check compilation debug logs** — Are there graph breaks or recompilation stalls?
9. **Check memory summary** — Are you close to VRAM limits or suffering fragmentation?

---

## 10. Expected Baseline Numbers (A100 80GB)

| Config | Tokens/sec | MFU | Step time |
|---|---|---|---|
| 124M, B=64, T=1024, `torch.compile` | ~30K–40K | 40–55 % | ~30–50 ms |
| 124M, B=64, T=1024, no `torch.compile` | ~15K–25K | 20–35 % | ~50–80 ms |
| 124M, B=32, T=512, `torch.compile` | ~10K–15K | 25–40 % | ~50–80 ms |
| 124M, B=8, T=128, `torch.compile` | ~2K–4K | 5–15 % | ~200–500 ms |

If your numbers are far below these, profiling will reveal the specific bottleneck.