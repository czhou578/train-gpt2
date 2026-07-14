"""
To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""
import os
import time
import math
import pickle
import logging
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from benchmark import MetricsLogger

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText / TinyStories
# I/O
out_dir = "out"
eval_interval = 1000
log_interval = 1
eval_iters = 30
eval_only = False
always_save_checkpoint = False
init_from = "scratch"  # 'scratch' | 'resume' | 'gpt2*'

# wandb
wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2"

# data
dataset = "tinystories"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# adamw
learning_rate = 6e-4
max_iters = 300
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = max(100, int(0.02 * max_iters))
lr_decay_iters = max_iters
min_lr = 6e-5

# DDP
backend = "nccl"

# system
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True

# sampling (used at checkpoints / end)
sample_max_new_tokens = 100
sample_start = "\n"
sample_temperature = 0.8
sample_top_k = 200

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Create a unique run directory so we never overwrite previous experiments
run_name = f"{dataset}_flashatt4_kernel_gpt2s_bs{batch_size}_ctx{block_size}_{time.strftime('%Y%m%d_%H%M%S')}"
run_dir = os.path.join(out_dir, run_name)
os.makedirs(run_dir, exist_ok=True)

metrics = MetricsLogger(run_dir)

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(run_dir, "train.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train")

# -----------------------------------------------------------------------------
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
logger.info(f"run directory: {run_dir}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data loader
if dataset == "tinystories_overfit":
    data_dir = os.path.join("data", "tinystories", "overfit")
else:
    data_dir = os.path.join("data", dataset)


def get_batch(split: str):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    logger.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == "scratch":
    logger.info("Initializing a new model from scratch")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    logger.info(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    logger.info(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

model.to(device)

num_params = model.get_num_params() if hasattr(model, "get_num_params") else sum(p.numel() for p in model.parameters())
logger.info(f"model has {num_params / 1e6:.2f}M parameters")

# log environment + run start
metrics.log_environment(dtype=dtype, device=device)
metrics.log_run_start(config=config, num_params=num_params)

# optimizer + scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

# compile
if compile:
    logger.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    compile_start = time.perf_counter()
    model = torch.compile(model)
    compile_time = time.perf_counter() - compile_start
    logger.info(f"compile time {compile_time:.2f}s")
    metrics.log_compile(compile_time)

# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch("val")
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

torch.cuda.reset_peak_memory_stats()
X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # ----- evaluation -----
    if iter_num % eval_interval == 0 and master_process:
        val_loss = estimate_loss()
        logger.info(f"step {iter_num}: val loss {val_loss:.4f}")
        metrics.log_eval(iter_num, val_loss.item(), lr)

        if wandb_log:
            wandb.log({"iter": iter_num, "val/loss": val_loss, "lr": lr, "mfu": running_mfu * 100})

        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                ckpt_path = os.path.join(run_dir, "ckpt.pt")
                logger.info(f"saving checkpoint to {ckpt_path}")
                torch.save(checkpoint, ckpt_path)
                metrics.log_checkpoint(iter_num, best_val_loss.item())

    if iter_num == 0 and eval_only:
        break

    # ----- forward / backward -----
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    # gradient clipping + capture norm
    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        metrics.log_optimizer(iter_num, grad_norm)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ----- timing & logging -----
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        tokens_per_sec = tokens_per_iter / dt
        samples_per_sec = batch_size / dt

        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        logger.info(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%, tok/s {tokens_per_sec:,.0f}"
        )

        metrics.log_train_step(
            iter_num=iter_num,
            loss=lossf,
            time_ms=dt * 1000,
            tokens_per_sec=tokens_per_sec,
            samples_per_sec=samples_per_sec,
            lr=lr,
            mfu_pct=running_mfu * 100 if running_mfu != -1.0 else None,
            grad_norm=grad_norm,
        )

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# final summary
metrics.write_summary(
    model="GPT2-small",
    dataset=dataset,
    iterations=max_iters,
    final_val_loss=best_val_loss,
    num_params=num_params,
)
logger.info(f"Training finished. All artifacts in: {run_dir}")