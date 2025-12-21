# TorchTitan Tutorial: Training LLMs from Scratch

This tutorial walks you through using TorchTitan to train language models,
starting from a single GPU with GPT-2 and scaling to multi-node training with
Qwen3. You'll learn:

1. How to install TorchTitan on a workstation with a single NVIDIA A100 GPU
2. How to add a new GPT-2 model to TorchTitan
3. How to train on Shakespeare data with WandB metrics visualization
4. How to scale to 8 B200 GPUs using DDP (Distributed Data Parallel)
5. How to evaluate trained models using lm_eval benchmarks
6. How to pretrain Qwen3 1.7B on 4 nodes (32 GPUs) with HSDP and ClimbMix dataset

---

## Prerequisites

- NVIDIA GPU with CUDA support (A100, H100, B200, or similar)
- Python 3.10+ (3.12 recommended)
- Basic familiarity with PyTorch and transformers

---

## Part 1: Installation

### 1.1 Clone TorchTitan

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
```

### 1.2 Install PyTorch Nightly

TorchTitan requires PyTorch nightly for the latest distributed training features:

```bash
# For CUDA 12.6 (recommended for A100)
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

# For CUDA 12.8 (recommended for B200/Blackwell)
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

### 1.3 Install TorchTitan and Dependencies

```bash
# Install TorchTitan in development mode
pip install -e .

# Install TorchAO for FP8 support (optional)
USE_CPP=0 pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126
```

### 1.4 Verify Installation

```bash
# Check PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Check TorchTitan
python -c "import torchtitan; print('TorchTitan installed successfully')"

# Quick validation (single GPU, fake backend)
NGPU=1 COMM_MODE=fake_backend python -m torchtitan.train \
    --job.config_file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --training.steps 1
```

---

## Part 2: Understanding TorchTitan Architecture

Before adding a new model, let's understand how TorchTitan organizes code:

```
torchtitan/
├── models/              # Production models (Llama3, DeepSeek, etc.)
├── experiments/         # Experimental models (including our GPT-2)
│   └── gpt2/
│       ├── model/
│       │   ├── args.py      # Model hyperparameters
│       │   └── model.py     # Model implementation
│       ├── infra/
│       │   └── parallelize.py  # Distributed training setup
│       ├── train_configs/
│       │   ├── debug_model.toml       # Single GPU config
│       │   └── gpt2_124m_openwebtext.toml  # Multi-GPU config
│       └── __init__.py      # TrainSpec registration
├── components/          # Reusable training components
├── hf_datasets/         # Dataset loading
└── train.py             # Main training loop
```

### Key Concepts

1. **TrainSpec**: Bundles model-specific components (model class, parallelization,
   optimizer, dataloader, etc.) so the generic trainer can work with any model.

2. **Model Protocol**: All models must implement `init_weights()` and accept
   `model_args` in `__init__()`.

3. **Parallelization Order**: TP → AC → compile → FSDP/DDP (order matters!)

---

## Part 3: The GPT-2 Model Implementation

We've created a GPT-2 implementation in `torchtitan/experiments/gpt2/`. Here's
how each component works:

### 3.1 Model Arguments (`model/args.py`)

```python
@dataclass
class GPT2ModelArgs(BaseModelArgs):
    dim: int = 768           # Hidden size
    n_layers: int = 12       # Number of transformer blocks
    n_heads: int = 12        # Number of attention heads
    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_seq_len: int = 1024  # Context window
    dropout: float = 0.0     # Dropout (0 for pretraining)
    bias: bool = True        # Use bias in linear layers
    weight_tying: bool = True  # Tie embedding and output weights
```

The `BaseModelArgs` requires two methods:
- `update_from_config()`: Update args from JobConfig (e.g., seq_len)
- `get_nparams_and_flops()`: Calculate model size and compute for metrics

### 3.2 Model Implementation (`model/model.py`)

The GPT-2 model follows the original architecture:

```
Input Tokens
    ↓
Token Embeddings + Positional Embeddings (learned)
    ↓
┌─────────────────────────────┐
│   Transformer Block × N     │
│   ├── LayerNorm             │
│   ├── Multi-Head Attention  │
│   ├── Residual Connection   │
│   ├── LayerNorm             │
│   ├── MLP (GELU)            │
│   └── Residual Connection   │
└─────────────────────────────┘
    ↓
LayerNorm
    ↓
Output Projection (weight-tied with embeddings)
    ↓
Logits
```

Key differences from Llama:
- **Learned positional embeddings** (not RoPE)
- **GELU activation** (not SwiGLU)
- **LayerNorm** (not RMSNorm)
- **Weight tying** between embeddings and output

### 3.3 Parallelization (`infra/parallelize.py`)

For this tutorial, we support:
- **Single GPU**: No parallelism
- **DDP**: Full model replica on each GPU, gradient synchronization
- **FSDP**: Sharded parameters for memory efficiency

```python
def parallelize_gpt2(model, parallel_dims, job_config):
    # Step 1: Activation Checkpointing (reduces memory)
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint, ...)

    # Step 2: torch.compile (optimizes computation)
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # Step 3: Data Parallelism
    if parallel_dims.fsdp_enabled:
        apply_fsdp(model, dp_mesh, ...)
    elif parallel_dims.dp_replicate_enabled:
        apply_ddp(model, dp_mesh, ...)
```

### 3.4 TrainSpec (`__init__.py`)

The TrainSpec connects everything:

```python
def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=GPT2Model,
        model_args=gpt2_configs,  # {"debugmodel": ..., "124M": ...}
        parallelize_fn=parallelize_gpt2,
        pipelining_fn=None,  # Not implemented for GPT-2
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
```

---

## Part 4: Training on Single GPU with Shakespeare

### 4.1 Download the GPT-2 Tokenizer

```bash
python scripts/download_hf_assets.py \
    --repo_id openai-community/gpt2 \
    --assets tokenizer \
    --local_dir ./assets/hf/gpt2
```

This downloads:
- `tokenizer.json` - The tokenizer definition
- `tokenizer_config.json` - BOS/EOS token configuration

### 4.2 Configuration File

The debug configuration (`torchtitan/experiments/gpt2/train_configs/debug_model.toml`):

```toml
[job]
dump_folder = "./outputs/gpt2_debug"
description = "GPT-2 debug training on tiny Shakespeare"

[model]
name = "gpt2"
flavor = "debugmodel"
hf_assets_path = "./assets/hf/gpt2"

[training]
local_batch_size = 8
seq_len = 1024
steps = 1000
dataset = "tiny_shakespeare"

[parallelism]
# Single GPU - no parallelism
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
```

### 4.3 Enable WandB Logging

Edit the config to enable WandB:

```toml
[metrics]
enable_tensorboard = false
enable_wandb = true
```

Or pass via command line:

```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/debug_model.toml" \
    ./run_train.sh --metrics.enable_wandb
```

Make sure you've logged into WandB:

```bash
wandb login
```

### 4.4 Run Training

```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/debug_model.toml" ./run_train.sh
```

Expected output:

```
[rank0] Starting training with config: GPT-2 debug training on tiny Shakespeare
[rank0] Model: GPT2Model with 4.2M parameters
[rank0] No data parallelism applied (single GPU mode)
[rank0] Step 1 | Loss: 10.82 | LR: 6.00e-06 | Tokens/s: 8192
[rank0] Step 2 | Loss: 10.45 | LR: 1.20e-05 | Tokens/s: 8543
...
```

### 4.5 View Metrics in WandB

Open your WandB dashboard to see:
- Training loss curve
- Learning rate schedule
- Tokens per second throughput
- GPU memory usage

---

## Part 5: Scaling to 8 GPUs with DDP

### 5.1 Understanding DDP vs FSDP

| Aspect | DDP | FSDP |
|--------|-----|------|
| **Memory** | Full model on each GPU | Model sharded across GPUs |
| **Communication** | Gradient all-reduce | All-gather + reduce-scatter |
| **Best for** | Models that fit in GPU memory | Large models |
| **Config** | `data_parallel_replicate_degree=8` | `data_parallel_shard_degree=8` |

For GPT-2 124M (~500MB), DDP is sufficient and simpler.

### 5.2 Configuration for 8 GPU DDP

The production configuration
(`torchtitan/experiments/gpt2/train_configs/gpt2_124m_openwebtext.toml`):

```toml
[model]
name = "gpt2"
flavor = "124M"
hf_assets_path = "./assets/hf/gpt2"

[optimizer]
name = "AdamW"
lr = 1.8e-3  # Higher LR for larger batch
weight_decay = 0.1
betas = [0.9, 0.95]

[lr_scheduler]
warmup_steps = 256
decay_ratio = 0.6  # 60% training, 40% cooldown
decay_type = "cosine"

[training]
local_batch_size = 64  # 64 * 8 GPUs = 512 global batch
seq_len = 1024
steps = 20345  # ~10B tokens
dataset = "openwebtext"

[parallelism]
# 8-way DDP (no FSDP)
data_parallel_replicate_degree = 8
data_parallel_shard_degree = 1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1

[compile]
enable = true
components = ["model"]
```

### 5.3 Run 8 GPU Training

```bash
# Ensure you have 8 GPUs available
CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/gpt2_124m_openwebtext.toml" ./run_train.sh
```

Or explicitly set NGPU:

```bash
NGPU=8 CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/gpt2_124m_openwebtext.toml" ./run_train.sh
```

### 5.4 Expected Training Metrics

With 8 B200 GPUs and the nanuGPT-style configuration:

| Metric | Expected Value |
|--------|----------------|
| Global batch size | 512 |
| Tokens per step | 524,288 |
| Total tokens | ~10B |
| Training time | ~2-3 hours |
| Final loss | ~2.8-3.0 |

---

## Part 6: Adding Your Own Dataset

### 6.1 Dataset Configuration

Datasets are defined in `torchtitan/hf_datasets/text_datasets.py`:

```python
DATASETS = {
    "openwebtext": DatasetConfig(
        path="sytelus/openwebtext",
        loader=_load_openwebtext_dataset,
        sample_processor=_process_openwebtext_text,
    ),
    "tiny_shakespeare": DatasetConfig(
        path="karpathy/tiny_shakespeare",
        loader=_load_tiny_shakespeare_dataset,
        sample_processor=_process_shakespeare_text,
    ),
}
```

### 6.2 Adding a Custom Dataset

1. **Define loader function**:

```python
def _load_my_dataset(dataset_path: str):
    return load_dataset(dataset_path, split="train", streaming=True)
```

2. **Define sample processor**:

```python
def _process_my_text(sample: dict[str, Any]) -> str:
    return sample["text"]  # Extract text field
```

3. **Add to DATASETS dict**:

```python
"my_dataset": DatasetConfig(
    path="my-org/my-dataset",
    loader=_load_my_dataset,
    sample_processor=_process_my_text,
),
```

4. **Use in config**:

```toml
[training]
dataset = "my_dataset"
```

---

## Part 7: Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size
--training.local_batch_size 4

# Enable activation checkpointing
--activation_checkpoint.mode selective

# Use FSDP instead of DDP
--parallelism.data_parallel_shard_degree 8 --parallelism.data_parallel_replicate_degree 1
```

**2. Tokenizer Not Found**

```bash
# Download GPT-2 tokenizer
python scripts/download_hf_assets.py \
    --repo_id openai-community/gpt2 \
    --assets tokenizer \
    --local_dir ./assets/hf/gpt2
```

**3. DDP Hangs on Startup**

```bash
# Check NCCL environment
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Try different NCCL settings
export NCCL_P2P_DISABLE=1
```

**4. Dataset Loading Slow**

```bash
# Increase dataloader workers
--training.dataloader.num_workers 4

# Enable pin_memory for GPU transfer
--training.dataloader.pin_memory true
```

---

## Part 8: Evaluation with lm_eval

After training your GPT-2 model, you'll want to evaluate it on standard benchmarks.
This section shows how to use EleutherAI's `lm_eval` framework for comprehensive
model evaluation.

### 8.1 Understanding the Evaluation Pipeline

The evaluation process involves three steps:

1. **Enable HuggingFace checkpoint export** during training
2. **Create a config.json** file for HuggingFace compatibility
3. **Run lm_eval** with appropriate benchmarks

```
TorchTitan Checkpoint (DCP format)
        ↓
    convert_to_hf.py (uses GPT2StateDictAdapter)
        ↓
HuggingFace Checkpoint (safetensors)
        ↓
    lm_eval with vLLM or HF backend
        ↓
Benchmark Results (HellaSwag, LAMBADA, etc.)
```

### 8.2 Configure Training for Checkpoint Export

Update your training config to save checkpoints in HuggingFace format:

```toml
[checkpoint]
enable_checkpoint = true
folder = "./outputs/gpt2_124m"
interval = 1000  # Save every 1000 steps

# Export to HuggingFace format on last save
last_save_in_hf = true
```

Or pass via command line:

```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/debug_model.toml" \
    ./run_train.sh \
    --checkpoint.enable_checkpoint \
    --checkpoint.folder "./outputs/gpt2_debug" \
    --checkpoint.interval 500 \
    --checkpoint.last_save_in_hf
```

### 8.3 Create GPT-2 config.json

The HuggingFace checkpoint requires a `config.json` file. Create one for your model:

**For debugmodel (4M params):**

```bash
cat > ./outputs/gpt2_debug/config.json << 'EOF'
{
  "architectures": ["GPT2LMHeadModel"],
  "model_type": "gpt2",
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_embd": 256,
  "n_layer": 4,
  "n_head": 4,
  "activation_function": "gelu_new",
  "resid_pdrop": 0.0,
  "embd_pdrop": 0.0,
  "attn_pdrop": 0.0,
  "layer_norm_epsilon": 1e-5,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "tie_word_embeddings": true,
  "torch_dtype": "float32"
}
EOF
```

**For GPT-2 124M:**

```bash
cat > ./outputs/gpt2_124m/config.json << 'EOF'
{
  "architectures": ["GPT2LMHeadModel"],
  "model_type": "gpt2",
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "activation_function": "gelu_new",
  "resid_pdrop": 0.0,
  "embd_pdrop": 0.0,
  "attn_pdrop": 0.0,
  "layer_norm_epsilon": 1e-5,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16"
}
EOF
```

### 8.4 Manual Checkpoint Conversion (Optional)

If you didn't use `last_save_in_hf`, you can convert checkpoints manually:

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
    --model gpt2 \
    --flavor 124M \
    --checkpoint_path ./outputs/gpt2_124m/step-20000 \
    --output_path ./outputs/gpt2_124m_hf
```

Then copy the config.json and tokenizer files:

```bash
cp ./outputs/gpt2_124m/config.json ./outputs/gpt2_124m_hf/
cp ./assets/hf/gpt2/tokenizer.json ./outputs/gpt2_124m_hf/
cp ./assets/hf/gpt2/tokenizer_config.json ./outputs/gpt2_124m_hf/
```

### 8.5 Set Up lm_eval Environment

**Important**: Installing `lm-eval` may break your TorchTitan environment due to
dependency conflicts. Create a separate environment:

```bash
# Create new environment for evaluation
conda create -n lm_eval python=3.11 -y
conda activate lm_eval

# Install lm-eval with vLLM backend (recommended for speed)
pip install "lm-eval[vllm]"

# Or install with HuggingFace backend only (simpler, slower)
pip install lm-eval
```

### 8.6 Run Evaluation Benchmarks

#### Recommended Benchmarks for GPT-2

For GPT-2 scale models, these benchmarks are most informative:

| Benchmark | Description | Metric | Shots |
|-----------|-------------|--------|-------|
| hellaswag | Commonsense reasoning | acc_norm | 0 |
| lambada_openai | Language modeling | acc | 0 |
| winogrande | Coreference resolution | acc | 0 |
| piqa | Physical intuition | acc | 0 |
| arc_easy | Science questions (easy) | acc | 0 |
| boolq | Boolean questions | acc | 0 |

#### Option A: Using vLLM Backend (Fast, GPU)

```bash
# Activate the lm_eval environment
conda activate lm_eval

# Run evaluation with vLLM (single GPU)
lm_eval --model vllm \
    --model_args pretrained=./outputs/gpt2_124m_hf,dtype=auto,gpu_memory_utilization=0.8 \
    --tasks hellaswag,lambada_openai,winogrande,piqa,arc_easy,boolq \
    --batch_size auto \
    --output_path ./outputs/gpt2_124m_eval

# For multi-GPU evaluation (8 GPUs)
lm_eval --model vllm \
    --model_args pretrained=./outputs/gpt2_124m_hf,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --tasks hellaswag,lambada_openai,winogrande,piqa,arc_easy,boolq \
    --batch_size auto \
    --output_path ./outputs/gpt2_124m_eval
```

#### Option B: Using HuggingFace Backend (Simpler, Slower)

```bash
conda activate lm_eval

# Run evaluation with HuggingFace transformers
lm_eval --model hf \
    --model_args pretrained=./outputs/gpt2_124m_hf \
    --tasks hellaswag,lambada_openai,winogrande,piqa,arc_easy,boolq \
    --batch_size 16 \
    --device cuda:0 \
    --output_path ./outputs/gpt2_124m_eval
```

### 8.7 Expected Results

For a well-trained GPT-2 124M model (~10B tokens on OpenWebText), expect results
similar to:

| Task | Metric | Expected Score |
|------|--------|----------------|
| hellaswag | acc_norm | 0.31 - 0.33 |
| lambada_openai | acc | 0.45 - 0.50 |
| winogrande | acc | 0.52 - 0.55 |
| piqa | acc | 0.65 - 0.70 |
| arc_easy | acc | 0.45 - 0.50 |
| boolq | acc | 0.60 - 0.65 |

**Note**: These are approximate values. Actual results depend on:
- Training data quality and quantity
- Training hyperparameters
- Random seed
- Number of training steps

A debug model trained on Shakespeare will score lower (near random baseline) on
these benchmarks since it was trained on a very small, domain-specific dataset.

### 8.8 Interpreting Results

**HellaSwag** (acc_norm ~0.32): Tests commonsense reasoning. GPT-2 124M is near
random baseline (0.25) but shows some learning.

**LAMBADA** (acc ~0.47): Tests long-range language modeling. Higher scores indicate
better context understanding.

**WinoGrande** (acc ~0.53): Tests coreference. Scores near 0.50 indicate near-random
performance (binary choice task).

**PIQA** (acc ~0.68): Tests physical intuition. GPT-2 124M typically performs well
here, above the 0.50 random baseline.

### 8.9 Running Periodic Evaluations

For longer training runs, you may want to evaluate at multiple checkpoints:

```bash
#!/bin/bash
# evaluate_checkpoints.sh

CHECKPOINT_DIR="./outputs/gpt2_124m"
OUTPUT_DIR="./outputs/gpt2_124m_eval"
CONFIG_JSON="./outputs/gpt2_124m/config.json"
TOKENIZER_DIR="./assets/hf/gpt2"

for step in 5000 10000 15000 20000; do
    STEP_DIR="${CHECKPOINT_DIR}/step-${step}"
    if [ -d "$STEP_DIR" ]; then
        echo "Evaluating step $step..."

        # Prepare HF checkpoint
        HF_DIR="${OUTPUT_DIR}/step-${step}"
        mkdir -p "$HF_DIR"

        # Convert if needed (or use existing HF export)
        python scripts/checkpoint_conversion/convert_to_hf.py \
            --model gpt2 \
            --flavor 124M \
            --checkpoint_path "$STEP_DIR" \
            --output_path "$HF_DIR"

        # Copy config and tokenizer
        cp "$CONFIG_JSON" "$HF_DIR/"
        cp "${TOKENIZER_DIR}/tokenizer.json" "$HF_DIR/"
        cp "${TOKENIZER_DIR}/tokenizer_config.json" "$HF_DIR/"

        # Run evaluation
        lm_eval --model vllm \
            --model_args pretrained="$HF_DIR",dtype=auto \
            --tasks hellaswag,lambada_openai \
            --batch_size auto \
            --output_path "${OUTPUT_DIR}/results-step-${step}"
    fi
done
```

### 8.10 Quick Smoke Test

For a quick sanity check, run a single fast benchmark:

```bash
# Quick test with LAMBADA (fast, ~14K examples)
lm_eval --model hf \
    --model_args pretrained=./outputs/gpt2_debug_hf \
    --tasks lambada_openai \
    --limit 100 \
    --batch_size 8 \
    --device cuda:0
```

This runs on just 100 examples and completes in seconds, useful for verifying
your checkpoint conversion worked correctly.

---

## Part 9: Multi-Node Pretraining with Qwen3 1.7B

This section walks through pretraining Qwen3 1.7B from scratch on 4 nodes of 8
B200 GPUs each (32 GPUs total) using HSDP (Hybrid Sharded Data Parallel) and
the ClimbMix dataset.

### 9.1 Prerequisites

#### Hardware Requirements

- 4 nodes with 8 NVIDIA B200 GPUs each (32 GPUs total)
- High-bandwidth inter-node networking (InfiniBand or AWS EFA recommended)
- At least 2TB shared storage for checkpoints

#### Software Requirements

- PyTorch nightly with CUDA 12.8 (for B200 support)
- TorchTitan installed on all nodes
- Slurm or similar job scheduler (optional but recommended)

### 9.2 Environment Setup

#### Docker Setup (Recommended)

For multi-node training, using Docker ensures consistent environments:

```bash
# Pull the PyTorch nightly container
docker pull nvcr.io/nvidia/pytorch:24.12-py3

# Or build a custom container
cat > Dockerfile << 'EOF'
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Install TorchTitan
WORKDIR /workspace
RUN git clone https://github.com/pytorch/torchtitan.git
WORKDIR /workspace/torchtitan
RUN pip install -e .

# Install additional dependencies
RUN pip install wandb datasets
EOF

docker build -t torchtitan:latest .
```

#### Native Installation (All Nodes)

If not using Docker, install on each node:

```bash
# On each node
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan

# Install PyTorch nightly for B200/Blackwell
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# Install TorchTitan
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### 9.3 Download Qwen3 Tokenizer

The model weights are trained from scratch, but we need the tokenizer:

```bash
# Download Qwen3 tokenizer (not the full model)
python scripts/download_hf_assets.py \
    --repo_id Qwen/Qwen3-1.7B \
    --assets tokenizer \
    --local_dir ./assets/hf/Qwen3-1.7B
```

This downloads:
- `tokenizer.json` - The tokenizer definition
- `tokenizer_config.json` - BOS/EOS token configuration
- `vocab.json` and `merges.txt` - BPE vocabulary files

### 9.4 WandB Setup

Configure Weights & Biases for metrics visualization:

```bash
# On the head node (or all nodes if using Slurm)
pip install wandb
wandb login

# Set project and entity (optional)
export WANDB_PROJECT="qwen3-pretraining"
export WANDB_ENTITY="your-team-name"
```

### 9.5 Understanding HSDP Configuration

HSDP (Hybrid Sharded Data Parallel) combines the best of FSDP and DDP:

```
┌─────────────────────────────────────────────────────────────┐
│                    HSDP Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Node 0 (FSDP Group 0)    Node 1 (FSDP Group 1)              │
│  ┌─────────────────┐      ┌─────────────────┐               │
│  │ GPU0 GPU1 GPU2  │      │ GPU0 GPU1 GPU2  │               │
│  │ GPU3 GPU4 GPU5  │ DDP  │ GPU3 GPU4 GPU5  │               │
│  │ GPU6 GPU7       │◄────►│ GPU6 GPU7       │               │
│  │                 │      │                 │               │
│  │ Sharded Params  │      │ Sharded Params  │               │
│  │ (high BW NVLink)│      │ (high BW NVLink)│               │
│  └─────────────────┘      └─────────────────┘               │
│           ▲                        ▲                         │
│           │      DDP Gradient      │                         │
│           │      Sync (EFA/IB)     │                         │
│           ▼                        ▼                         │
│  Node 2 (FSDP Group 2)    Node 3 (FSDP Group 3)              │
│  ┌─────────────────┐      ┌─────────────────┐               │
│  │ GPU0-7 Sharded  │ DDP  │ GPU0-7 Sharded  │               │
│  │                 │◄────►│                 │               │
│  └─────────────────┘      └─────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Configuration for 32 GPUs (4 nodes × 8 GPUs):
- data_parallel_shard_degree = 8   (FSDP within each node)
- data_parallel_replicate_degree = 4 (DDP across 4 nodes)
```

**Why HSDP?**
- FSDP within nodes uses high-bandwidth NVLink (900 GB/s on B200)
- DDP across nodes uses lower-bandwidth network (400 Gb/s EFA/IB)
- This topology matches the hardware hierarchy for optimal performance

### 9.6 Training Configuration

The optimized configuration file is at:
`torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml`

#### Configuration with Explanations

```toml
[model]
name = "qwen3"
flavor = "1.7B"
hf_assets_path = "./assets/hf/Qwen3-1.7B"
```

**Model Selection Rationale:**
- **Qwen3 1.7B** is chosen as a balance between training time and model capability
- Large enough to learn meaningful representations (~1.7B parameters)
- Small enough to train efficiently on 32 GPUs without excessive parallelism complexity
- Uses the Qwen3 architecture: RoPE, GQA (16 heads, 8 KV heads), SwiGLU activation

---

```toml
[optimizer]
name = "AdamW"
lr = 3e-4
eps = 1e-8
fused = true
```

**Optimizer Rationale:**
- **lr = 3e-4**: Standard learning rate for 1-2B parameter models. Based on scaling laws:
  - Smaller models (100M-500M) often use 6e-4 to 1e-3
  - Larger models (7B+) typically use 1e-4 to 3e-4
  - 3e-4 is the "Chinchilla optimal" range for this model size
- **fused = true**: Uses CUDA fused AdamW kernel, ~20% faster than standard AdamW
- **eps = 1e-8**: Default epsilon for numerical stability

---

```toml
[lr_scheduler]
warmup_steps = 2000
decay_ratio = 0.1
decay_type = "cosine"
```

**LR Scheduler Rationale:**
- **warmup_steps = 2000**: ~1% of total training (200K steps)
  - Warmup prevents early training instability with large batch sizes
  - Rule of thumb: 0.5-2% of total steps for warmup
  - With 32 GPUs and large global batch, warmup is critical
- **decay_ratio = 0.1**: Final LR is 10% of peak (3e-5)
  - Allows model to converge to sharper minima at end of training
  - Values between 0.0-0.1 are standard
- **decay_type = "cosine"**: Smooth decay, widely used for LLM pretraining
  - Better than linear decay for long training runs
  - Matches Llama, GPT-3, and other foundation model recipes

---

```toml
[training]
local_batch_size = 8
seq_len = 4096
max_norm = 1.0
steps = 200000
dataset = "climbmix"
mixed_precision = "bfloat16"
```

**Training Hyperparameters Rationale:**

- **local_batch_size = 8**: Per-GPU batch size
  - Global batch = 8 × 32 GPUs = 256 sequences
  - Tokens per step = 256 × 4096 = **1,048,576 tokens (~1M)**
  - B200 with 192GB HBM3 can handle batch size 8-16 for 1.7B model
  - Larger batches improve GPU utilization but may hurt convergence

- **seq_len = 4096**: Qwen3's default context length
  - Matches the model's pre-configured `max_seq_len`
  - Longer sequences (8K+) would require more memory or context parallelism

- **max_norm = 1.0**: Gradient clipping threshold
  - Standard value for LLM training, prevents gradient explosions
  - Values 0.5-1.0 are typical; lower values (0.5) for unstable training

- **steps = 200000**: Total training steps
  - Total tokens = 200K steps × 1M tokens/step = **200B tokens**
  - ClimbMix has ~400B tokens, so we train on ~50% of the dataset
  - Chinchilla-optimal for 1.7B model is ~34B tokens (20× params)
  - We overtrain 6× for better downstream performance

- **mixed_precision = "bfloat16"**: BF16 mixed precision training
  - BF16 preferred over FP16 for training stability (larger dynamic range)
  - B200 GPUs have excellent BF16 performance (2.25 PFLOPS)
  - Reduces memory by ~50% compared to FP32

---

```toml
[parallelism]
data_parallel_replicate_degree = 4
data_parallel_shard_degree = 8
fsdp_reshard_after_forward = "default"
tensor_parallel_degree = 1
context_parallel_degree = 1
pipeline_parallel_degree = 1
```

**Parallelism Strategy Rationale:**

- **data_parallel_shard_degree = 8**: FSDP within each node
  - Shards model parameters across 8 GPUs per node
  - Uses NVLink (900 GB/s on B200) for all-gather/reduce-scatter
  - Memory per GPU: ~1.7B params × 2 bytes / 8 = ~425MB model weights
  - With optimizer states (8 bytes/param): ~1.7GB per GPU

- **data_parallel_replicate_degree = 4**: DDP across 4 nodes
  - Each node has a complete sharded replica
  - Gradient sync uses inter-node network (400 Gb/s EFA/IB)
  - DDP communication is just gradient all-reduce (lighter than FSDP)

- **Why HSDP over pure FSDP?**
  - Pure FSDP (dp_shard=32) would require all-gather across nodes
  - Inter-node bandwidth is 10-20× slower than intra-node NVLink
  - HSDP minimizes cross-node communication

- **tensor_parallel_degree = 1**: TP disabled
  - Qwen3 1.7B fits comfortably in memory without TP
  - TP adds communication overhead, only needed for very large models (70B+)

- **context_parallel_degree = 1**: CP disabled
  - Not needed for 4096 sequence length
  - CP is for very long sequences (32K+) that don't fit in memory

---

```toml
[checkpoint]
enable = true
folder = "checkpoint"
interval = 5000
last_save_model_only = false
export_dtype = "bfloat16"
async_mode = "async"
last_save_in_hf = true
```

**Checkpointing Rationale:**

- **interval = 5000**: Save every 5000 steps (~1.5-2 hours of training)
  - Balance between checkpoint overhead and recovery granularity
  - At 200K total steps, this creates ~40 checkpoints
  - If training crashes, maximum loss is 5000 steps

- **async_mode = "async"**: Non-blocking checkpoint saves
  - Training continues while checkpoint is written to disk
  - Reduces checkpoint overhead from minutes to near-zero
  - Requires sufficient CPU memory to buffer checkpoint

- **export_dtype = "bfloat16"**: Save weights in BF16
  - Matches training precision, no conversion loss
  - Smaller checkpoint files than FP32

- **last_save_in_hf = true**: Auto-export final checkpoint
  - Converts to HuggingFace safetensors format
  - Ready for inference or lm_eval without manual conversion

---

```toml
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
```

**Activation Checkpointing Rationale:**

- **mode = "selective"**: Checkpoint only high-memory operations
  - Recomputes attention and FFN activations during backward pass
  - Reduces memory by ~30-40% with minimal compute overhead (~10%)
  - "full" mode saves more memory but adds ~30% compute overhead

- **selective_ac_option = "op"**: Operation-based selection
  - Automatically selects which ops to checkpoint based on memory/compute tradeoff
  - Alternative: "int" = checkpoint every N layers

---

```toml
[compile]
enable = true
components = ["model", "loss"]
```

**Compilation Rationale:**

- **enable = true**: Use torch.compile for kernel fusion
  - Fuses operations like LayerNorm + attention into optimized kernels
  - Typically 10-30% speedup for transformer models
  - First few steps are slower due to compilation (one-time cost)

- **components = ["model", "loss"]**: What to compile
  - Compiles the model forward pass and loss computation
  - Optimizer is not compiled (already uses fused kernels)

---

#### Summary: Key Numbers

| Parameter | Value | Calculation/Reasoning |
|-----------|-------|----------------------|
| Global batch size | 256 | 8 per GPU × 32 GPUs |
| Tokens per step | 1,048,576 | 256 × 4096 |
| Total tokens | 200B | 200K steps × 1M tokens |
| Peak LR | 3e-4 | Standard for 1-2B models |
| Final LR | 3e-5 | 10% of peak (decay_ratio=0.1) |
| Warmup tokens | 2B | 2000 steps × 1M tokens |
| Memory per GPU | ~60-80GB | Model + optimizer + activations |
| Checkpoint size | ~6.8GB | 1.7B × 2 bytes × 2 (model+optimizer) |

### 9.7 Launch Multi-Node Training

#### Option A: Using Slurm

Create or modify `multinode_trainer.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3_1.7b_pretrain
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --output=logs/qwen3_%j.log

# Get head node IP
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node ($head_node_ip)"

# NCCL settings for optimal performance
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152

# For AWS with EFA
export FI_PROVIDER="efa"
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

# WandB settings
export WANDB_PROJECT="qwen3-pretraining"

# Memory settings
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Launch training
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml"

srun torchrun \
    --nnodes 4 \
    --nproc_per_node 8 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.train \
    --job.config_file ${CONFIG_FILE}
```

Submit the job:

```bash
sbatch multinode_trainer.slurm
```

#### Option B: Manual Launch (Without Slurm)

On the head node (Node 0):

```bash
# Set environment variables
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
export WORLD_SIZE=32
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# On Node 0
torchrun \
    --nnodes 4 \
    --nproc_per_node 8 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    -m torchtitan.train \
    --job.config_file ./torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml
```

On other nodes (run simultaneously):

```bash
# On Node 1
export MASTER_ADDR=<head_node_ip>
torchrun --nnodes 4 --nproc_per_node 8 --node_rank 1 \
    --master_addr $MASTER_ADDR --master_port 29500 \
    -m torchtitan.train \
    --job.config_file ./torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml

# On Node 2
torchrun --nnodes 4 --nproc_per_node 8 --node_rank 2 ...

# On Node 3
torchrun --nnodes 4 --nproc_per_node 8 --node_rank 3 ...
```

### 9.8 Monitoring Training Metrics

#### WandB Dashboard

Once training starts, view metrics at https://wandb.ai:

| Metric | Description |
|--------|-------------|
| `loss` | Training loss (should decrease) |
| `learning_rate` | Current LR (warmup → peak → decay) |
| `tokens_per_second` | Training throughput |
| `tokens_per_second_per_gpu` | Per-GPU efficiency |
| `mfu` | Model FLOPS Utilization (% of peak) |
| `memory/max_active_pct` | GPU memory usage |

#### Expected Metrics for Qwen3 1.7B on 32 B200 GPUs

| Metric | Expected Value |
|--------|----------------|
| Tokens per second | ~1.5-2M tokens/s |
| Tokens per GPU per second | ~50K tokens/s |
| MFU | 40-50% |
| Memory usage | ~60-70% of 192GB |
| Time to 200K steps | ~3-4 days |

#### Terminal Logging

TorchTitan logs metrics to the terminal:

```
[rank0] Step 100 | Loss: 8.42 | LR: 3.00e-05 | Tokens/s: 1,523,456 | MFU: 45.2%
[rank0] Step 200 | Loss: 7.15 | LR: 6.00e-05 | Tokens/s: 1,612,892 | MFU: 47.1%
...
```

### 9.9 Performance Tuning

#### Tune Batch Size

If you have memory headroom, increase batch size:

```bash
# Try larger batch size
--training.local_batch_size 16
```

#### Enable FP8 (H100/B200)

For additional speedup on Hopper/Blackwell GPUs:

```toml
[model]
converters = ["float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
```

#### Activation Checkpointing

If OOM, use full activation checkpointing:

```bash
--activation_checkpoint.mode full
```

#### Disable Compilation (Debugging)

If you see compilation issues:

```bash
--compile.enable=false
```

### 9.10 Checkpointing and Recovery

#### Checkpoint Structure

Checkpoints are saved to `./outputs/qwen3_1.7b_climbmix/checkpoint/step-XXXXX/`:

```
step-5000/
├── __0_0.distcp           # Distributed checkpoint shards
├── __1_0.distcp
├── ...
├── .metadata              # Checkpoint metadata
└── train_state.json       # Training state (step, optimizer)
```

#### Resume from Checkpoint

To resume training:

```bash
torchrun ... -m torchtitan.train \
    --job.config_file ./torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml \
    --checkpoint.resume
```

#### Export to HuggingFace

The final checkpoint is auto-exported to HuggingFace format. To manually convert:

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
    --model qwen3 \
    --flavor 1.7B \
    --checkpoint_path ./outputs/qwen3_1.7b_climbmix/checkpoint/step-200000 \
    --output_path ./outputs/qwen3_1.7b_hf
```

### 9.11 Troubleshooting

#### NCCL Timeout

```bash
# Increase timeout
export NCCL_TIMEOUT=1800

# Debug NCCL issues
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
```

#### OOM Errors

```bash
# Reduce batch size
--training.local_batch_size 4

# Enable full activation checkpointing
--activation_checkpoint.mode full

# Disable compilation (uses less memory)
--compile.enable=false
```

#### Slow Inter-Node Communication

```bash
# For AWS EFA
export FI_PROVIDER="efa"
export FI_EFA_USE_HUGE_PAGE=0

# For InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

#### Dataset Streaming Issues

```bash
# Increase dataloader workers
--training.dataloader.num_workers 8

# Enable pin memory
--training.dataloader.pin_memory true
```

### 9.12 Switching from HSDP to Pure FSDP

This section explains how to switch from HSDP (Hybrid Sharded Data Parallel) to
pure FSDP (Fully Sharded Data Parallel) for Qwen3 1.7B training.

#### When to Use FSDP Instead of HSDP

| Scenario | Recommendation |
|----------|----------------|
| High-bandwidth inter-node network (800+ Gb/s) | FSDP may be faster |
| Single node training | FSDP (HSDP not applicable) |
| Maximum memory efficiency needed | FSDP shards across all GPUs |
| Simpler configuration | FSDP has fewer parameters |
| Lower inter-node bandwidth (<400 Gb/s) | **Keep HSDP** |
| 4+ nodes with standard networking | **Keep HSDP** |

#### HSDP vs FSDP Comparison

```
HSDP (Current Configuration)          Pure FSDP
┌──────────────────────────────┐      ┌──────────────────────────────┐
│  dp_replicate=4, dp_shard=8  │      │  dp_replicate=1, dp_shard=32 │
├──────────────────────────────┤      ├──────────────────────────────┤
│                              │      │                              │
│  Node 0    Node 1            │      │  All 32 GPUs in one          │
│  [8 GPUs]  [8 GPUs]          │      │  FSDP group                  │
│     ↕         ↕              │      │                              │
│   FSDP      FSDP             │      │  GPU 0 ←→ GPU 1 ←→ ... ←→ 31 │
│     ↕         ↕              │      │                              │
│  Node 2    Node 3            │      │  All-gather and reduce-      │
│  [8 GPUs]  [8 GPUs]          │      │  scatter across ALL GPUs     │
│                              │      │  (including cross-node)      │
│  DDP sync across nodes       │      │                              │
│  (gradient all-reduce only)  │      │                              │
└──────────────────────────────┘      └──────────────────────────────┘

Communication:                         Communication:
- FSDP: intra-node (NVLink)           - FSDP: all GPUs (includes network)
- DDP: inter-node (gradients only)    - More cross-node traffic
```

#### Step 1: Understand the Configuration Change

The key change is in the `[parallelism]` section:

```toml
# HSDP Configuration (current)
[parallelism]
data_parallel_replicate_degree = 4    # 4 replicas (DDP across nodes)
data_parallel_shard_degree = 8        # 8-way sharding (FSDP within nodes)
# Product: 4 × 8 = 32 GPUs

# Pure FSDP Configuration (target)
[parallelism]
data_parallel_replicate_degree = 1    # No replication
data_parallel_shard_degree = 32       # 32-way sharding (FSDP across all GPUs)
# Product: 1 × 32 = 32 GPUs
```

#### Step 2: Create FSDP Configuration File

Create a new config file or modify the existing one:

```bash
# Copy the HSDP config
cp torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml \
   torchtitan/models/qwen3/train_configs/qwen3_1.7b_fsdp_climbmix.toml
```

Edit the new file and change the parallelism section:

```toml
[parallelism]
# Pure FSDP: shard across all 32 GPUs
data_parallel_replicate_degree = 1    # No replication (changed from 4)
data_parallel_shard_degree = -1       # Auto-calculate: uses all 32 GPUs
fsdp_reshard_after_forward = "default"
tensor_parallel_degree = 1
context_parallel_degree = 1
pipeline_parallel_degree = 1
```

**Note:** Setting `data_parallel_shard_degree = -1` automatically calculates the
shard degree based on available GPUs (32 in this case).

#### Step 3: Adjust for Memory and Performance

With pure FSDP, you may need to adjust other settings:

```toml
[training]
# FSDP may allow slightly larger batch sizes due to better memory distribution
local_batch_size = 8    # Can try 10-12 with pure FSDP

[activation_checkpoint]
# May need more aggressive checkpointing with FSDP
mode = "selective"      # Keep as-is, or use "full" if OOM
```

#### Step 4: Launch Training with FSDP

Using the new config file:

```bash
# With Slurm
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_1.7b_fsdp_climbmix.toml"
sbatch multinode_trainer.slurm

# Or via command-line override (without creating new file)
torchrun --nnodes 4 --nproc_per_node 8 ... \
    --job.config_file ./torchtitan/models/qwen3/train_configs/qwen3_1.7b_hsdp_climbmix.toml \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.data_parallel_shard_degree -1
```

#### Step 5: Compare Performance

Run both configurations and compare metrics in WandB:

| Metric | HSDP Expected | FSDP Expected | Notes |
|--------|---------------|---------------|-------|
| Tokens/s | 1.5-2M | 1.2-1.8M | FSDP may be slower with standard networking |
| Memory/GPU | 60-80GB | 50-70GB | FSDP may use less memory |
| MFU | 40-50% | 35-45% | HSDP typically has better MFU |
| Startup time | Faster | Slower | FSDP needs to shard across network |

#### When to Choose Each Option

**Choose HSDP when:**
- You have 4+ nodes with standard inter-node networking
- Maximum throughput is the priority
- Network bandwidth is limited (< 400 Gb/s)

**Choose FSDP when:**
- You have very high-bandwidth networking (NVLink across nodes, 800+ Gb/s)
- You need maximum memory efficiency (larger models)
- Simpler configuration is preferred
- Single-node training (HSDP not applicable)

#### Quick Reference: Common Configurations

```toml
# Single Node (8 GPUs) - Pure FSDP only option
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 8

# 2 Nodes (16 GPUs) - HSDP
data_parallel_replicate_degree = 2
data_parallel_shard_degree = 8

# 2 Nodes (16 GPUs) - Pure FSDP
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 16

# 4 Nodes (32 GPUs) - HSDP (recommended)
data_parallel_replicate_degree = 4
data_parallel_shard_degree = 8

# 4 Nodes (32 GPUs) - Pure FSDP
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 32

# 8 Nodes (64 GPUs) - HSDP
data_parallel_replicate_degree = 8
data_parallel_shard_degree = 8

# 8 Nodes (64 GPUs) - Alternative HSDP (larger shards)
data_parallel_replicate_degree = 4
data_parallel_shard_degree = 16
```

---

## Part 10: Next Steps

Now that you've trained models with TorchTitan, explore:

1. **Tensor Parallelism**: See `torchtitan/models/llama3/infra/parallelize.py`
   for TP implementation

2. **Pipeline Parallelism**: See `torchtitan/distributed/pipeline_parallel.py`

3. **FP8 Training**: Add `--model.converters="quantize.linear.float8"` for
   H100/B200 GPUs with even faster training

4. **Larger Models**: Try Llama 3 70B or 405B with the production configs:
   ```bash
   CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_70b.toml" ./run_train.sh
   ```

5. **MoE Models**: Train Qwen3-MoE or DeepSeek-V3 with expert parallelism

6. **Custom Models**: Use the GPT-2 experiment as a template for your own models

7. **In-Training Validation**: Use the `Validator` class for validation during
   training (see `docs/evaluation.md`)

---

## Summary

In this tutorial, you learned:

- ✅ How to install TorchTitan for NVIDIA GPUs
- ✅ How TorchTitan organizes model code (TrainSpec, ModelProtocol)
- ✅ How to train GPT-2 on a single GPU with Shakespeare
- ✅ How to visualize training with WandB
- ✅ How to scale to 8 GPUs with DDP
- ✅ How to add custom datasets
- ✅ How to evaluate models using lm_eval benchmarks
- ✅ How to convert checkpoints for HuggingFace compatibility
- ✅ How to pretrain Qwen3 1.7B on multi-node clusters with HSDP
- ✅ How to configure and monitor large-scale distributed training

TorchTitan provides a clean, modular framework for distributed LLM training.
The patterns you learned here apply to any model architecture!
