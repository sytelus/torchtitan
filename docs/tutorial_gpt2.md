# TorchTitan Tutorial: Training GPT-2 from Scratch

This tutorial walks you through using TorchTitan to train a GPT-2 language model,
starting from a single GPU and scaling to multi-GPU training. You'll learn:

1. How to install TorchTitan on a workstation with a single NVIDIA A100 GPU
2. How to add a new GPT-2 model to TorchTitan
3. How to train on Shakespeare data with WandB metrics visualization
4. How to scale to 8 B200 GPUs using DDP (Distributed Data Parallel)

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

## Part 8: Next Steps

Now that you've trained GPT-2 with TorchTitan, explore:

1. **Tensor Parallelism**: See `torchtitan/models/llama3/infra/parallelize.py`
   for TP implementation

2. **Pipeline Parallelism**: See `torchtitan/distributed/pipeline_parallel.py`

3. **FP8 Training**: Add `--model.converters="quantize.linear.float8"` for
   H100/B200 GPUs

4. **Larger Models**: Try Llama 3 8B with the production configs:
   ```bash
   CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
   ```

5. **Custom Models**: Use the GPT-2 experiment as a template for your own models

---

## Summary

In this tutorial, you learned:

- ✅ How to install TorchTitan for NVIDIA GPUs
- ✅ How TorchTitan organizes model code (TrainSpec, ModelProtocol)
- ✅ How to train GPT-2 on a single GPU with Shakespeare
- ✅ How to visualize training with WandB
- ✅ How to scale to 8 GPUs with DDP
- ✅ How to add custom datasets

TorchTitan provides a clean, modular framework for distributed LLM training.
The patterns you learned here apply to any model architecture!
