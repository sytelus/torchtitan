# Long Context Extension in TorchTitan

This tutorial covers how to extend context length for Llama-style models in TorchTitan
using RoPE (Rotary Position Embeddings) scaling. Long context extension allows models
trained on shorter sequences to generalize to much longer contexts.

---

## Overview

TorchTitan supports long context extension through two complementary mechanisms:

1. **RoPE Scaling** - Interpolates position embeddings to support longer sequences
2. **Context Parallelism (CP)** - Distributes long sequences across multiple GPUs

This guide focuses on RoPE scaling configuration. For Context Parallelism, see
[parallelism_primer.md](parallelism_primer.md).

---

## How RoPE Scaling Works

Standard RoPE encodes positions using rotation frequencies. When extending beyond
the original training context, naive position extrapolation fails. RoPE scaling
solves this by:

1. **Scaling low frequencies** - Stretches long-wavelength components to cover
   extended positions
2. **Preserving high frequencies** - Keeps short-wavelength (local attention)
   patterns intact
3. **Smooth interpolation** - Blends the two regimes for medium frequencies

This is the Llama 3.1 RoPE scaling approach, which empirically works better than
simple linear interpolation.

---

## RoPEScalingArgs Configuration

The `RoPEScalingArgs` dataclass controls RoPE scaling behavior:

```python
from dataclasses import dataclass

@dataclass
class RoPEScalingArgs:
    scaling_factor: float = 8.0
    """Overall RoPE scaling factor (higher enables longer contexts)."""

    low_freq_factor: float = 1.0
    """Scaling factor for low-frequency components."""

    high_freq_factor: float = 4.0
    """Scaling factor for high-frequency components."""

    original_max_position_embeddings: int = 8192
    """Base context length the model was trained with."""
```

### Parameter Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scaling_factor` | 8.0 | Multiplier for position scaling. `8.0` means 8× context extension (e.g., 8K → 64K) |
| `low_freq_factor` | 1.0 | Controls the wavelength threshold for "low frequency" bands |
| `high_freq_factor` | 4.0 | Controls the wavelength threshold for "high frequency" bands |
| `original_max_position_embeddings` | 8192 | The context length the base model was trained on |

### The Math Behind It

For each frequency band in RoPE:

```
wavelength = 2π / frequency

if wavelength < (original_max_pos / high_freq_factor):
    # High frequency: no scaling (preserve local attention)
    scaled_freq = freq

elif wavelength > (original_max_pos / low_freq_factor):
    # Low frequency: full scaling (stretch for long range)
    scaled_freq = freq / scaling_factor

else:
    # Medium frequency: smooth interpolation
    scaled_freq = interpolate(freq, freq / scaling_factor)
```

---

## Step-by-Step: Extending Llama 3 8B to 64K Context

### Step 1: Create a Custom Model Flavor

Create a new model flavor with RoPE scaling configured for your target context length.

**Option A: Add to existing model flavors**

Edit `torchtitan/models/llama3/__init__.py`:

```python
from .model.args import TransformerModelArgs, RoPEScalingArgs

llama3_args = {
    # ... existing flavors ...

    # New: 8B with 64K context extension
    "8B_64k": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=65536,  # 64K target context
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=8.0,                      # 8× extension (8K → 64K)
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,   # Llama 3's base context
        ),
    ),
}
```

**Option B: Create an experiments folder**

For cleaner separation, create a new experiment:

```
torchtitan/experiments/llama3_longcontext/
├── __init__.py
└── train_configs/
    └── llama3_8b_64k.toml
```

```python
# torchtitan/experiments/llama3_longcontext/__init__.py
from torchtitan.models.llama3 import get_train_spec as get_base_spec
from torchtitan.models.llama3.model.args import TransformerModelArgs, RoPEScalingArgs

# Extended context flavors
llama3_longcontext_args = {
    "8B_64k": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=65536,
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "8B_128k": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=131072,
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=16.0,                     # 16× extension (8K → 128K)
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
}

def get_train_spec():
    spec = get_base_spec()
    # Override model args with long context versions
    spec = spec._replace(model_args=llama3_longcontext_args)
    return spec
```

### Step 2: Create the Training Configuration

Create a TOML config for long context training:

```toml
# torchtitan/models/llama3/train_configs/llama3_8b_64k.toml
# Long context extension: Llama 3 8B with 64K context
# Recommended: 8+ H100/B200 GPUs with Context Parallelism

[job]
dump_folder = "./outputs/llama3_8b_64k"
description = "Llama 3 8B long context fine-tuning (64K)"

[profiling]
enable_profiling = false
save_traces_folder = "profile_trace"
profile_freq = 500

[metrics]
log_freq = 10
enable_tensorboard = true
save_tb_folder = "tb"

[model]
name = "llama3"
flavor = "8B_64k"                              # Our new 64K flavor
hf_assets_path = "./assets/hf/Llama-3.1-8B"

[optimizer]
name = "AdamW"
lr = 1e-4                                      # Lower LR for fine-tuning
eps = 1e-8
fused = true

[lr_scheduler]
warmup_steps = 500
decay_ratio = 0.1
decay_type = "cosine"

[training]
local_batch_size = 1                           # Reduce for memory (long sequences)
seq_len = 65536                                # 64K sequence length
max_norm = 1.0
steps = 10000
dataset = "c4"                                 # Or your long-context dataset
mixed_precision = "bfloat16"

[parallelism]
# Context Parallelism is CRITICAL for 64K sequences
# CP=8 splits 64K into 8×8K chunks
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1                # FSDP uses remaining GPUs
context_parallel_degree = 8                    # Split sequence across 8 GPUs
tensor_parallel_degree = 1
pipeline_parallel_degree = 1

[checkpoint]
enable = true
folder = "checkpoint"
interval = 1000
export_dtype = "bfloat16"
async_mode = "async"

[activation_checkpoint]
mode = "selective"                             # Essential for long context
selective_ac_option = "op"

[compile]
enable = true                                  # Recommended for performance
components = ["model", "loss"]
```

### Step 3: Prepare Your Long-Context Dataset

Long context training requires documents that actually use the extended context.
Short documents padded to 64K won't help the model learn long-range dependencies.

**Good datasets for long context:**
- Long-form articles, books, legal documents
- Multi-turn conversations
- Code repositories with full file context
- Scientific papers with full references

**Dataset configuration:**

```toml
[training]
dataset = "your_long_context_dataset"
seq_len = 65536
```

### Step 4: Run Training

```bash
# Single node with 8 GPUs (CP=8)
torchrun --nproc_per_node=8 \
    -m torchtitan.train \
    --job.config_file torchtitan/models/llama3/train_configs/llama3_8b_64k.toml

# Multi-node (2 nodes × 8 GPUs = 16 GPUs, CP=8 + FSDP=2)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    -m torchtitan.train \
    --job.config_file torchtitan/models/llama3/train_configs/llama3_8b_64k.toml
```

---

## Scaling Factor Guidelines

| Target Context | Original Context | Scaling Factor | Notes |
|----------------|------------------|----------------|-------|
| 16K | 8K | 2.0 | Modest extension |
| 32K | 8K | 4.0 | Common target |
| 64K | 8K | 8.0 | Llama 3.1 default |
| 128K | 8K | 16.0 | Requires significant compute |
| 256K | 8K | 32.0 | Experimental, needs careful tuning |

**Rule of thumb:** `scaling_factor = target_context / original_context`

---

## Combining with Context Parallelism

For very long contexts (32K+), combine RoPE scaling with Context Parallelism:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    64K Sequence with CP=8                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GPU 0      GPU 1      GPU 2      GPU 3      GPU 4      GPU 5  ...  │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐         │
│  │ 8K  │   │ 8K  │   │ 8K  │   │ 8K  │   │ 8K  │   │ 8K  │  ...    │
│  │chunk│   │chunk│   │chunk│   │chunk│   │chunk│   │chunk│         │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘         │
│                                                                      │
│  RoPE Scaling: positions 0-8K mapped to 0-64K frequency space       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Important constraints:**

1. **CP requires SDPA attention** - Flex and Varlen attention don't support CP yet
2. **Sequence length must be divisible by `tp × 2 × cp`**
3. **Memory scales with context** - Use activation checkpointing

```toml
[parallelism]
context_parallel_degree = 8        # Sequence parallelism
data_parallel_shard_degree = 4     # FSDP for model sharding

[activation_checkpoint]
mode = "selective"                 # Critical for memory
selective_ac_option = "op"
```

---

## Memory Optimization for Long Context

Long sequences consume significantly more memory. Use these techniques:

### 1. Activation Checkpointing

```toml
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"    # Operation-level checkpointing
```

For extreme memory pressure:

```toml
[activation_checkpoint]
mode = "full"                 # Checkpoint every layer (slower but saves more memory)
```

### 2. FP8/MXFP8 Quantization

On H100:
```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
filter_fqns = ["output"]
```

On B200:
```toml
[model]
converters = ["quantize.linear.mx"]

[quantize.linear.mx]
recipe_name = "mxfp8_cublas"
filter_fqns = ["output"]
```

Note: Standard B200 GPUs work with `pip install --pre torchao`. GB200 (NVLink-connected
B200 pairs) requires building TorchAO from source.

### 3. Reduce Batch Size

```toml
[training]
local_batch_size = 1          # Minimum batch size per GPU
```

### 4. Gradient Accumulation

To maintain effective batch size with small per-GPU batches:

```toml
[training]
local_batch_size = 1
gradient_accumulation_steps = 8   # Effective batch = 8 per GPU
```

---

## Fine-tuning vs. Continued Pre-training

### Fine-tuning (Recommended for most cases)

- Lower learning rate: `1e-5` to `1e-4`
- Fewer steps: 1K-10K
- Use long-context instruction data

```toml
[optimizer]
lr = 1e-4

[training]
steps = 5000
```

### Continued Pre-training

- Higher learning rate: `1e-4` to `3e-4`
- More steps: 10K-100K+
- Use diverse long-form text data

```toml
[optimizer]
lr = 3e-4

[training]
steps = 50000
```

---

## Verifying Long Context Capability

After training, verify the model can use extended context:

1. **Perplexity at different positions** - Should remain stable across the full context
2. **Needle-in-haystack test** - Hide information at various positions, test retrieval
3. **Long-range dependency tasks** - Summarization, QA over long documents

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM on long sequences | Memory scales with seq_len² | Use CP, activation checkpointing, reduce batch |
| Loss spikes | Position encoding instability | Lower LR, use warmup, check scaling_factor |
| Poor long-range retrieval | Insufficient training | More steps, better data, verify RoPE config |
| CP errors | Wrong attention type | Use `attn_type="sdpa"` (default) |
| Slow training | Missing compilation | Enable `compile = true` |

---

## Complete Example: Llama 3 8B → 128K Context

Here's a complete configuration for extending Llama 3 8B to 128K context:

**Model flavor** (add to `torchtitan/models/llama3/__init__.py`):

```python
"8B_128k": TransformerModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    max_seq_len=131072,
    rope_scaling_args=RoPEScalingArgs(
        scaling_factor=16.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
),
```

**Training config** (`llama3_8b_128k.toml`):

```toml
[job]
dump_folder = "./outputs/llama3_8b_128k"
description = "Llama 3 8B 128K context (4 nodes × 8 GPUs)"

[model]
name = "llama3"
flavor = "8B_128k"
hf_assets_path = "./assets/hf/Llama-3.1-8B"
converters = ["quantize.linear.float8"]

[optimizer]
name = "AdamW"
lr = 5e-5
fused = true

[lr_scheduler]
warmup_steps = 1000
decay_type = "cosine"

[training]
local_batch_size = 1
seq_len = 131072
max_norm = 1.0
steps = 20000
dataset = "long_context_dataset"
mixed_precision = "bfloat16"

[parallelism]
data_parallel_shard_degree = 4
context_parallel_degree = 8

[activation_checkpoint]
mode = "full"

[compile]
enable = true
components = ["model", "loss"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
filter_fqns = ["output"]
```

**Launch command:**

```bash
# 4 nodes × 8 H100 GPUs = 32 GPUs
# CP=8 (sequence), FSDP=4 (model sharding)
torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    -m torchtitan.train \
    --job.config_file torchtitan/models/llama3/train_configs/llama3_8b_128k.toml
```

---

## References

- [Llama 3.1 Paper](https://arxiv.org/abs/2407.21783) - RoPE scaling methodology
- [RoFormer Paper](https://arxiv.org/abs/2104.09864) - Original RoPE formulation
- [YaRN Paper](https://arxiv.org/abs/2309.00071) - Alternative scaling approaches
- [docs/parallelism_primer.md](parallelism_primer.md) - Context Parallelism details
- [docs/modelling.md](modelling.md) - Model architecture and attention types
