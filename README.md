# train-gpt2

Train a GPT-2 (124M) language model from scratch on a single NVIDIA DGX Spark, and measure every part of the training step.

[![CI](https://github.com/czhou578/train-gpt2/actions/workflows/ci.yml/badge.svg)](https://github.com/czhou578/train-gpt2/actions/workflows/ci.yml)

The code starts from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. This repository adds:

- **Structured metrics.** Every training step writes one JSON record with the loss, the throughput, the model FLOPs utilization (MFU), the GPU memory, and a timing breakdown of forward, backward, data load, and optimizer.
- **A context-length sweep driver.** It writes one self-contained run directory per experiment.
- **Attention kernel benchmarks** on the DGX Spark: PyTorch math backend, PyTorch SDPA FlashAttention, and FlashAttention-4.
- **A profiling guide** for GPU training on this hardware.
- **Results and loss curves** for TinyStories and WikiText-103.

## Contents

| Path | Purpose |
|---|---|
| [model.py](model.py) | The GPT model: `GPTConfig`, `GPT`, causal self-attention, optimizer setup, MFU estimate, and sampling. |
| [train.py](train.py) | Single-GPU training loop with structured logging and a per-step timing breakdown. |
| [train_optimization.py](train_optimization.py) | Context-length sweep. Trains one model per context length and writes a run directory for each. |
| [benchmark.py](benchmark.py) | `MetricsLogger`. Writes `metrics.jsonl`, `config.json`, and `summary.json` for a run. Reads GPU telemetry through NVML. |
| [sample.py](sample.py) | Generates text from a checkpoint or from OpenAI GPT-2 weights. |
| [download_datasets.py](download_datasets.py) | Downloads TinyStories or WikiText-103 from the Hugging Face Hub. |
| [prepare.py](prepare.py) | Tokenizes a dataset with the GPT-2 tokenizer into `data/<dataset>/{train,val,test}.bin`. |
| [train_gpt2.py](train_gpt2.py) | A minimal one-file GPT-2 loop on Tiny Shakespeare (`input.txt`), after Karpathy's "Let's reproduce GPT-2". |
| [tests/](tests/) | CPU smoke tests for `model.py`. |
| [docs/profiling_guide.md](docs/profiling_guide.md) | How to find the bottleneck: nvidia-smi, MFU, torch.profiler, nsys, ncu, and memory. |
| [docs/roadmap.md](docs/roadmap.md) | The original project roadmap and working notes. |
| [out/](out/) | Logs, metrics, and loss curves from the runs in the Results section. |

## Hardware and software

All results come from one machine.

| Item | Value |
|---|---|
| Machine | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Grace Blackwell), 128 GB unified memory |
| CPU | 20 Arm cores |
| Driver | 580.173.02 |
| PyTorch | 2.13.0, CUDA 13.0 build |
| Python | 3.12 |
| Precision | bfloat16 autocast, TF32 matmul |

The MFU values in this repository use the A100 bfloat16 peak of 312 TFLOPS as the reference, as nanoGPT does. The GB10 has a different peak. Treat MFU as a relative number between runs on this machine.

## Setup

1. Create a virtual environment with Python 3.12 and activate it.
2. Install PyTorch from the wheel index that matches your CUDA version. The DGX Spark uses CUDA 13:
   ```bash
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
   ```
   For a machine without a GPU, use `--index-url https://download.pytorch.org/whl/cpu`. The CPU build runs the tests only.
3. Install the remaining packages:
   ```bash
   pip install -r requirements.txt
   ```

Optional packages:

| Package | Needed for |
|---|---|
| `transformers` | `init_from = 'gpt2*'` in `train.py` and `sample.py`. It loads the OpenAI weights. |
| `wandb` | `wandb_log = True` in `train.py`. |
| `flash-attn` 4 (`flash_attn.cute`) | `train_gpt2.py`, and the FlashAttention-4 variant of `model.py`. |

## Data

The training scripts read flat `uint16` token arrays from `data/<dataset>/`. The arrays are not in git. Build them in two steps.

Step 1. Download the raw text:
```bash
python download_datasets.py tinystories     # eminorhan/tinystories, config 10M_1
python download_datasets.py wikitext-103    # Salesforce/wikitext, config wikitext-103-raw-v1
```

Step 2. Tokenize:
```bash
python prepare.py tinystories               # -> data/tinystories/{train,val,test}.bin
python prepare.py wikitext-103              # -> data/wikitext-103/{train,val,test}.bin
```

`prepare.py` uses the GPT-2 tokenizer from `tiktoken` and appends the end-of-text token after each document. WikiText-103 keeps its official splits, without empty lines. TinyStories has one split. The script shuffles it with seed 42 and cuts it 90/5/5.

To build the small overfit set that `dataset = 'tinystories_overfit'` expects:
```bash
python prepare.py tinystories --max_examples 20000 --out_dir data/tinystories/overfit
```

| Dataset | Train tokens | Size on disk |
|---|---:|---:|
| TinyStories 10M | 11.2M | 22 MB |
| WikiText-103 | 119M | 238 MB |

## Training

`train.py` has no command-line parser. Edit the configuration block at the top of the file, then run:
```bash
python train.py
```

The settings that matter most:

| Setting | Default | Meaning |
|---|---|---|
| `dataset` | `wikitext-103` | Folder under `data/`. `tinystories_overfit` maps to `data/tinystories/overfit`. |
| `batch_size`, `gradient_accumulation_steps`, `block_size` | 64, 2, 128 | Tokens per iteration is the product of the three. |
| `n_layer`, `n_head`, `n_embd` | 12, 12, 768 | GPT-2 small, 124M parameters. |
| `learning_rate`, `max_iters`, `warmup_iters` | 3e-4, 2000, 100 | Cosine decay with a linear warmup. |
| `compile` | `True` | `torch.compile` roughly doubles the throughput. |
| `metrics_log_file`, `train_log_file` | `metrics_wikitext_128.jsonl`, `train_wikitext_128.log` | Output names. Change them for each run, because the script appends to existing files. |

The script writes into `out/`:

- `<train_log_file>`: the console log.
- `<metrics_log_file>`: one JSON object per line. The schema is below.
- `ckpt.pt`: the checkpoint with the best validation loss. Git ignores it.

### Metrics schema

| `event` | Fields |
|---|---|
| `run_start` | `config` (every setting), `num_params` |
| `eval` | `iter`, `val_loss`, `lr`, `gpu_util_pct`, `mem_allocated_gb`, `mem_reserved_gb`, `max_mem_allocated_gb`, `max_mem_reserved_gb` |
| `checkpoint` | `iter`, `best_val_loss` |
| `train_step` | `iter`, `loss`, `time_ms`, `fwd_time_ms`, `bwd_time_ms`, `load_time_ms`, `opt_time_ms`, `mfu_pct`, `tokens_per_sec`, `lr` |

Every record also has a `timestamp` in Unix seconds. The four timing fields come from CUDA events, so they add one synchronization per step instead of one per phase.

### Context-length sweep

```bash
python train_optimization.py
```

Edit `context_lengths` and the configuration block at the top. The script writes one run directory per context length:

```
out/<dataset>_flashatt4_ctx<block_size>_bs<batch_size>_<timestamp>/
├── config.json      every setting of the run
├── metrics.jsonl    environment, run_start, train_step, eval, optimizer, checkpoint, and summary records
├── summary.json     best val loss, peak memory, average tokens/s, wall time
└── train.log        console log
```

With `nvidia-ml-py` installed, `benchmark.py` also records GPU utilization, power, temperature, and clocks every 10 steps.

### Sampling

```bash
python sample.py
```

Edit `init_from`, `out_dir`, `start`, `num_samples`, and `max_new_tokens` at the top of the file.

### Minimal reference loop

`train_gpt2.py` is a one-file version of the whole pipeline on Tiny Shakespeare. It needs `flash-attn` 4. Read it first if you are new to GPT training. The results below do not use it.

## Results

All runs use GPT-2 small (124M parameters, 12 layers, 768 wide), AdamW, bfloat16, `torch.compile`, and one GB10 GPU.

### Context length on WikiText-103

2000 iterations, batch 64 with 2 accumulation steps, peak learning rate 3e-4, weight decay 0.1.

| Context | Tokens / iter | Best val loss | Tokens / s | MFU |
|---:|---:|---:|---:|---:|
| 128 | 16,384 | 4.28 | 47k | 11.4% |
| 256 | 32,768 | 3.92 | 50k | 12.3% |
| 512 | 65,536 | 3.64 | 52k | 13.2% |
| 1024 | 131,072 | 3.45 | 53k | 14.5% |

Tokens/s and MFU are the median over the last 200 steps. A longer context sees more tokens per iteration, so it reaches a lower loss in the same number of steps. The throughput stays almost flat. The smaller contexts do not saturate the GPU.

Logs: `out/train_wikitext_<ctx>.log` and `out/metrics_wikitext_<ctx>.jsonl`. Loss curves: `out/loss_curve_wikitext128.html`, `out/loss_curve_wikitext256.html`, `out/wikitext_512_train_val_loss_curve.html`, and `out/wikitext_train_val_loss_curve_1024.html`. The HTML files load Plotly from a CDN.

### Context length on TinyStories

300 iterations, batch 32, no accumulation, peak learning rate 6e-4. Each run has its own `out/tinystories_flashatt4_ctx*` directory.

| Context | Best val loss | Avg tokens / s | Peak memory | Wall time |
|---:|---:|---:|---:|---:|
| 128 | 3.16 | 38.4k | 4.4 GB | 56 s |
| 256 | 3.00 | 44.5k | 8.5 GB | 84 s |
| 512 | 2.89 | 48.7k | 13.8 GB | 134 s |
| 1024 | 2.95 | 50.7k | 24.3 GB | 234 s |
| 2048 | 2.96 | 38.9k | 65.0 GB | 561 s |

The throughput peaks at a context of 1024. At 2048 the attention cost and the 65 GB of activations slow the step down. Memory grows almost linearly with the context.

### Attention kernels on TinyStories

Context 1024, batch 32, 300 iterations. The three runs differ only in the attention call inside `model.py`. These runs evaluated the validation loss at step 0 only, so the table reports throughput and memory.

| Kernel | Avg tokens / s | Peak memory | Run directory |
|---|---:|---:|---|
| PyTorch SDPA, math backend | 51.3k | 22.8 GB | `out/tinystories_math_kernel_*` |
| PyTorch SDPA, FlashAttention backend | 51.5k | 22.8 GB | `out/tinystories_flashatt_kernel_*` |
| FlashAttention-4 (`flash_attn.cute`) | 51.2k | 22.8 GB | `out/tinystories_flashatt4_kernel_*` |

The three runs give the same throughput and the same peak memory. Two explanations fit this result. Either the step is not attention-bound at 124M parameters and a context of 1024, or the backend selection did not take effect under `torch.compile`. The math backend normally needs several extra gigabytes for the attention matrix, so the second explanation deserves a check. Confirm the active kernel with `torch.profiler` before you draw a conclusion. [docs/profiling_guide.md](docs/profiling_guide.md) shows how.

### Long baseline runs

| Run | Batch | Iterations | Tokens seen | Best val loss | Avg tokens / s | Files |
|---|---|---:|---:|---:|---:|---|
| TinyStories, large batch | 12 × 40 accumulation, ctx 1024 | 2000 | 983M (about 88 epochs) | 2.74 | 50.8k | `out/train_baseline.log`, `out/metrics_tinystories_baseline.jsonl`, `out/tinystories_baseline_loss-curve.html` |
| TinyStories, batch 32 | 32, ctx 1024 | 2000 | 66M (about 6 epochs) | 1.58 | 50.2k | `out/tinystories_flash_baseline_*` |

The first run repeats the 11M-token dataset about 88 times and overfits. Its train loss falls to 0.25 while its val loss stays at 2.74. The second run sees each token about six times and reaches a val loss of 1.58. Sample generations from early runs are in [docs/roadmap.md](docs/roadmap.md).

## Profiling

[docs/profiling_guide.md](docs/profiling_guide.md) explains how to find the bottleneck of a training step on this machine: nvidia-smi utilization, MFU, the per-step timing breakdown that `train.py` logs, `torch.profiler`, Nsight Systems, Nsight Compute, and memory snapshots.

## Tests and continuous integration

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests build a tiny GPT on the CPU. They check the shapes, the loss, the gradients, the optimizer groups, generation, and the block-size crop. GitHub Actions runs them on every push and pull request. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Dependency updates

Dependabot checks `requirements*.txt` and the GitHub Actions versions every Monday. See [.github/dependabot.yml](.github/dependabot.yml). Security updates are enabled in the repository settings.

[.github/workflows/dependabot-auto-merge.yml](.github/workflows/dependabot-auto-merge.yml) handles the pull requests that Dependabot opens:

1. It runs the CI smoke test on the pull request.
2. If the test passes and the update is a patch or minor version bump, it squash-merges the pull request.
3. Major version bumps stay open for a human review.

Security fixes almost always ship as patch or minor releases, so they merge without manual action. To merge every Dependabot pull request, including major bumps, delete the `if:` condition on the merge step.

## Roadmap

[docs/roadmap.md](docs/roadmap.md) holds the original plan: dataset and model scaling, a CUDA kernel roadmap (LayerNorm, GELU, softmax, matmul, FlashAttention), and the experiment matrix. Done so far: the TinyStories and WikiText-103 baselines, the context-length sweeps, the attention-kernel comparison, the metrics pipeline, and the profiling guide.

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) and the lecture ["Let's reproduce GPT-2"](https://github.com/karpathy/build-nanogpt) by Andrej Karpathy. `model.py`, `train.py`, and `sample.py` started from nanoGPT.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019).
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) and [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023).
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023), through the `eminorhan/tinystories` dataset.
- [WikiText-103](https://arxiv.org/abs/1609.07843) (Merity et al., 2016), through `Salesforce/wikitext`.

## License

MIT. See [LICENSE](LICENSE). The nanoGPT portions keep their original MIT copyright notice.
