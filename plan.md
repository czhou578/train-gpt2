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



Overfit a set of tiny stories (20k samples)

# model
n_layer = 2
n_head = 4
n_embd = 128
block_size = 128

# training
batch_size = 16
learning_rate = 1e-3
max_iters = 2000

# dropout
dropout = 0.0

step 2000: train loss 2.3538, val loss 2.4013
saving checkpoint to out
iter 2000: loss 2.3536, time 1660.57ms, mfu 2.69%

python sample.py \
    --out_dir=out \
    --dataset=tinystories10m \
    --num_samples=5 \
    --max_new_tokens=200

After that, the big dog came to the park. The dog loved to play and play in the park. The dog saw Sue and Sue under the bed. Sue was scared, but she did not help.
Sue closed the bushes and said, "I want to play with you!" They played and laughed and had fun. When the dog got to Tom, they decided to play together. Sue was a nice girl. They played a lotion with Sue, and they played together.<|endoftext|>Once upon a time, there was a girl named Lucy. She had a friend named Jerry. Tom loved to play with Sue and Sue. They had fun computer all day long. They would laugh and enjoy their dance.
One day, Tom's mom came to play with all the little ones. But Tom hurt and did not mind. Tom was sad. He thought, "I am, Tom. I love to feel better."
Tom tried to use the computer, but he could not help. Tom did not want to play with Lily. So, Tom tried to balance on the computer. He was not happy to take the computer with him. He tried to push the computer, but he was difficult. Tom was sad, and he did not want to hurt Sara.
Tom tried to climb the computer, but he hurt his knee. He fell down and started to worry. The computer did not move. The computer made Tim's face and came back.
Sara and Tom were sad. They did not want to go. They did not know what to do. They said they were sorry and lost. They had a bad ending.<|endoftext|>Once upon a time, there was a little dog named Spot. Spot lived in a small house with many toys. One day, Tim went to the store with his mom. They were on the street.
Tim's mom saw grown-up, and said, "Lily, can we have enough food with this crayons!" They looked at each other and saw a big, red bucket. They were curious and excited.
"Look, the box is in the grass!" Tim said. He wanted to pick the box and open it, but it was difficult to get it. So, he started to break it. He pulled and pulled, and the box.
"Oh no, what are you doing?" Lily asked.
Lily said, "We need to cut the box, Lily."
"Give it