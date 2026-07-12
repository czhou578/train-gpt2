# GPT-2 Reproduction Roadmap on a Single DGX Spark

## Goals
- Learn modern LLM pretraining from first principles.
- Reproduce the engineering ideas behind GPT-2 rather than the exact compute budget.
- Benchmark every stage of training.
- Eventually replace core PyTorch operations with custom CUDA kernels.

## Recommended Dataset Progression

| Phase | Dataset | Target Tokens | Purpose |
|---|---|---:|---|
| 0 | TinyStories | 10–50M | Verify pipeline |
| 1 | WikiText-103 | 50–100M | Debug training/evaluation |
| 2 | FineWeb-Edu (subset) | 100M | First real pretraining |
| 3 | FineWeb-Edu (subset) | 250M | Scaling experiment |
| 4 | FineWeb-Edu (subset) | 500M | Larger-scale training |
| 5 | FineWeb-Edu (subset) | 1B | Overnight training |

## Model Scaling

| Model | Approx. Params | Goal |
|---|---:|---|
| Nano | 20M | Fast iteration |
| Small | 50–80M | Hyperparameter exploration |
| GPT-2 Small | 124M | Main reproduction |

## Experiment Matrix

### 1. Dataset Scaling
Train the same model on:
- 100M tokens
- 250M tokens
- 500M tokens
- 1B tokens

Record:
- Train loss
- Validation loss
- Perplexity
- Tokens/sec
- GPU memory
- Wall-clock time
- Sample generations

### 2. Model Scaling
Keep the dataset fixed (e.g. 250M tokens).

Compare:
- 20M parameters
- 50–80M parameters
- 124M parameters

Measure:
- Final validation loss
- Throughput
- Memory usage
- Convergence speed

### 3. Context Length
Train with:
- 256
- 512
- 1024
- 2048 tokens

Measure:
- Memory
- Throughput
- Quality

### 4. Batch Size
Compare several global batch sizes while keeping effective tokens constant using gradient accumulation if needed.

Measure:
- Stability
- Tokens/sec
- Time to target loss

### 5. Learning Rate
Try:
- Cosine decay
- Warmup lengths
- Peak LR sweep

### 6. Optimizer
Compare:
- AdamW
- Lion
- Adafactor

### 7. Precision
Evaluate:
- BF16
- FP16
- FP8 (if supported)

Record throughput and convergence.

### 8. Weight Decay
Sweep multiple values and compare validation loss.

## CUDA Kernel Roadmap

Replace one operation at a time.

1. LayerNorm
2. RMSNorm
3. GELU
4. Softmax
5. Matrix multiplication (learning project)
6. FlashAttention
7. Fused kernels
8. KV-cache operations

For every replacement:
- Verify numerical correctness.
- Benchmark kernel latency.
- Measure end-to-end training speed.

## Benchmark Dashboard

Collect:
- Tokens/sec
- Examples/sec
- Time/epoch
- GPU utilization
- GPU memory
- CPU utilization
- Power consumption
- Validation loss
- Perplexity
- Checkpoint size

## Suggested Timeline

### Week 1
- Training pipeline
- TinyStories
- Nano GPT

### Week 2
- FineWeb-Edu 100M
- Hyperparameter tuning

### Week 3
- GPT-2 Small on 250–500M tokens

### Week 4
- CUDA LayerNorm + GELU

### Week 5
- FlashAttention implementation

### Week 6
- Full benchmark report and inference runtime

## Stretch Goals

- Build a GPT inference engine.
- Implement paged KV cache.
- Continuous batching.
- Mini vLLM.
- Train with your own CUDA kernels where practical.
- Publish benchmark reports comparing PyTorch vs. custom kernels.