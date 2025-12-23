# Model Architecture and Configuration in TorchTitan

This guide covers model architecture choices, attention implementations, initialization
strategies, and optimization techniques specific to TorchTitan.

---

## Supported Models

TorchTitan includes implementations of several model architectures:

| Model | Name | Flavors |
|-------|------|---------|
| Llama 3/3.1 | `llama3` | `debugmodel`, `8B`, `8B_flex`, `8B_varlen`, `70B`, `405B` |
| Llama 4 | `llama4` | `Scout-17B-16E`, `Scout-17B-16E-Instruct`, `Maverick-17B-128E` |
| Qwen3 | `qwen3` | `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B` |
| DeepSeek-V3 | `deepseek_v3` | `671B` |
| Flux | `flux` | `dev`, `schnell` |

### Configuration

```toml
[model]
name = "llama3"           # Model architecture
flavor = "8B"             # Model size/variant
hf_assets_path = "./assets/hf/Llama3.1-8B"  # Tokenizer location
```

---

## Attention Implementations

TorchTitan supports three attention implementations, configured via `attn_type` in model
flavors. Each has different capabilities for masking, performance, and parallelism.

### 1. SDPA (Scaled Dot-Product Attention) - Default

```python
attn_type = "sdpa"  # Default for standard flavors (8B, 70B, etc.)
```

**Implementation:** `torch.nn.functional.scaled_dot_product_attention`

**How it works:**
- Uses PyTorch's native SDPA with automatic backend selection
- Tries backends in priority order: cuDNN -> Flash Attention -> Efficient Attention -> Math
- Simple causal masking via `is_causal=True` flag

**When to use:**
- Standard pre-training without document masking
- When using Context Parallelism (CP) - **required**
- Best hardware compatibility

---

### 2. Flex Attention

```python
attn_type = "flex"  # Model flavors: 8B_flex, 70B_flex
attn_mask_type = "block_causal"  # Enable document masking
```

**Implementation:** `torch.nn.attention.flex_attention` (PyTorch 2.5+)

**How it works:**
- Uses `BlockMask` with customizable mask modifier functions
- Supports arbitrary attention patterns via composable masks
- Compiled with `max-autotune` for Triton kernel optimization

**Available mask modifiers:**

| Mask | Function | Description |
|------|----------|-------------|
| Causal | `get_causal_mask_mod()` | Standard causal (attend to past only) |
| Document | `get_document_mask_mod(batch, eos_id)` | Block cross-document attention |
| Sliding Window | `get_sliding_window_mask_mod(window_size)` | Local attention window |
| Fixed Block | `get_fixed_block_mask_mod(block_size)` | Block-wise attention |

**When to use:**
- Document masking to prevent cross-document attention
- Sliding window attention for efficiency
- Custom attention patterns

---

### 3. Varlen Attention

```python
attn_type = "varlen"  # Model flavors: 8B_varlen, 70B_varlen
attn_mask_type = "block_causal"
```

**Implementation:** `torch.nn.attention.varlen.varlen_attn`

**How it works:**
- Designed for variable-length sequences within a batch
- Uses cumulative sequence lengths (`cu_seqlens`) to track document boundaries
- Packs sequences without explicit padding, handles variable lengths natively

**Key data structure:**
```python
VarlenMetadata(
    cu_seq_q,   # Cumulative query positions [0, doc1_len, doc1+doc2_len, ...]
    cu_seq_k,   # Cumulative key positions
    max_q,      # Maximum query length in batch
    max_k,      # Maximum key length in batch
)
```

**When to use:**
- Variable-length documents with document masking
- Flash Attention v2 compatible workloads

---

### Attention Type Comparison

| Feature | SDPA | Flex | Varlen |
|---------|------|------|--------|
| **Default** | Yes | | |
| **Custom masks** | No | Yes | Limited |
| **Document masking** | No | Yes | Yes |
| **Sliding window** | No | Yes | No |
| **Context Parallelism** | Yes | No | No |
| **Requires compile** | No | Yes | Yes |
| **Model flavors** | `8B`, `70B` | `8B_flex` | `8B_varlen` |

### Important Constraint

**Context Parallelism (CP) only works with SDPA.** If you set `context_parallel_degree > 1`
with flex or varlen attention, training will fail:

```python
# This raises NotImplementedError
if cp_degree > 1 and attn_type != "sdpa":
    raise NotImplementedError("CP support for FlexAttention is still in progress.")
```

---

## Model Initialization

TorchTitan uses a custom initialization scheme that differs significantly from PyTorch
defaults. Understanding this is crucial for training stability and adding new layers.

### What is Truncated Normal?

TorchTitan uses **truncated normal initialization** (`nn.init.trunc_normal_`) instead
of PyTorch's default Kaiming uniform. A truncated normal distribution:

1. Samples from a normal distribution with specified mean and std
2. **Truncates** (discards and resamples) values outside `[a, b]` bounds
3. Prevents extreme outlier weights that can destabilize training

```python
# TorchTitan's truncated normal (default bounds: a=-2*std, b=2*std)
nn.init.trunc_normal_(weight, mean=0.0, std=0.02)

# With explicit bounds (used for output projection)
nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
```

**Why truncated normal?**
- Normal distribution can produce extreme values (e.g., 5+ standard deviations)
- These outliers can cause exploding gradients in deep networks
- Truncation keeps all weights within a bounded range
- Empirically better for transformer training stability

### TorchTitan vs PyTorch Default Initialization

| Layer Type | PyTorch Default | TorchTitan |
|------------|-----------------|------------|
| `nn.Linear` | Kaiming Uniform: `U(-1/√in, 1/√in)` | Truncated Normal: `N(0, 0.02)` or depth-scaled |
| `nn.Embedding` | Normal: `N(0, 1)` | Normal: `N(0, 1)` (same) |
| `nn.RMSNorm` | Ones for weight | `reset_parameters()` (ones) |
| Bias terms | Uniform: `U(-1/√in, 1/√in)` | Not used (most layers have `bias=False`) |

**Key difference:** PyTorch's Kaiming init scales with layer width (`1/√fan_in`),
while TorchTitan uses a fixed small std (0.02) with depth-based scaling for residual
layers.

### Where Initialization Happens in Code

TorchTitan uses a **two-phase initialization** pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Initialization Flow                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase 1: Meta Device              Phase 2: Materialization                 │
│   (No memory allocated)             (Weights initialized)                    │
│                                                                              │
│   with torch.device("meta"):        model.to_empty(device=gpu)               │
│       model = Transformer(args)     model.init_weights()                     │
│                                                                              │
│   ↓                                 ↓                                        │
│   Model structure created           Storage allocated, then                  │
│   but no tensor storage             init_weights() fills values              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code locations:**

1. **Model construction on meta device** (`train.py:408-412`):
   ```python
   with torch.device("meta"):
       model = train_spec.model_cls(model_args)
   ```

2. **Materialization and initialization** (`train.py:597-603`):
   ```python
   model.to_empty(device=init_device)  # Allocate storage
   with torch.no_grad():
       model.init_weights(buffer_device=buffer_device)  # Initialize values
   ```

3. **Model's init_weights method** (`models/llama3/model/model.py:457-492`):
   ```python
   def init_weights(self, buffer_device=None):
       # Initialize embeddings
       nn.init.normal_(self.tok_embeddings.weight)

       # Initialize each layer
       for layer in self.layers.values():
           layer.init_weights()

       # Initialize output projection
       nn.init.trunc_normal_(self.output.weight, mean=0.0, std=dim**-0.5)
   ```

### Depth-Aware Initialization

When `depth_init=True` (default), residual pathway layers use depth-scaled std:

```python
# In TransformerBlock.__init__
if model_args.depth_init:
    # Earlier layers: larger std, later layers: smaller std
    self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
else:
    # All layers same std
    self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5
```

**Example for 32-layer model:**

| Layer | depth_init=True | depth_init=False |
|-------|-----------------|------------------|
| 0 | 0.0141 | 0.0025 |
| 1 | 0.0100 | 0.0025 |
| 15 | 0.0035 | 0.0025 |
| 31 | 0.0025 | 0.0025 |

### Initialization by Component

| Component | Init Method | Standard Deviation | Code Location |
|-----------|-------------|-------------------|---------------|
| Embeddings | Normal | 1.0 | `Transformer.init_weights` |
| Attention Q/K/V | Truncated Normal | 0.02 | `Attention.init_weights` |
| Attention Output (Wo) | Truncated Normal | depth-scaled | `Attention.init_weights` |
| FFN Gate (W1) | Truncated Normal | 0.02 | `FeedForward.init_weights` |
| FFN Down (W2) | Truncated Normal | depth-scaled | `FeedForward.init_weights` |
| FFN Up (W3) | Truncated Normal | depth-scaled | `FeedForward.init_weights` |
| Output Projection | Truncated Normal | `dim^(-0.5)` | `Transformer.init_weights` |
| RMSNorm | Ones | N/A | `reset_parameters()` |

### How TorchTitan Handles Unknown Layers

TorchTitan does **not** automatically initialize layers. Each module must explicitly
define an `init_weights()` method. If you add a new layer type:

**Option 1: Layer uses PyTorch defaults (risky)**

If a layer doesn't have `init_weights()` called, it keeps whatever values were set
during `to_empty()` - which is **uninitialized garbage** on meta device!

```python
# BAD: This layer won't be initialized properly
class MyCustomLayer(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(512, 512)
    # No init_weights() method!
```

**Option 2: Implement init_weights() (recommended)**

```python
# GOOD: Explicit initialization
class MyCustomLayer(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(512, 512, bias=False)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.linear.weight, mean=0.0, std=init_std)
```

**Option 3: Use reset_parameters() for simple cases**

For layers that should use PyTorch defaults, call `reset_parameters()`:

```python
def init_weights(self):
    self.my_layer.reset_parameters()  # Uses PyTorch's default init
```

### Overriding Initialization

#### Method 1: Custom init_weights in Your Model

Create a custom model class that overrides `init_weights()`:

```python
from torchtitan.models.llama3.model.model import Transformer

class MyTransformer(Transformer):
    def init_weights(self, buffer_device=None):
        # Call parent initialization
        super().init_weights(buffer_device)

        # Override specific layers
        for layer in self.layers.values():
            # Use Xavier instead of truncated normal for attention
            nn.init.xavier_uniform_(layer.attention.wq.weight)
            nn.init.xavier_uniform_(layer.attention.wk.weight)
            nn.init.xavier_uniform_(layer.attention.wv.weight)
```

#### Method 2: Post-initialization Hook

Apply custom initialization after the standard init:

```python
def custom_init(model):
    """Apply custom initialization after model.init_weights()"""
    for name, param in model.named_parameters():
        if "output.weight" in name:
            # Custom init for output layer
            nn.init.xavier_normal_(param, gain=0.01)

# In your training script, after init_weights()
model.init_weights()
custom_init(model)
```

#### Method 3: Custom TrainSpec

For complete control, create a custom TrainSpec with your model class:

```python
# torchtitan/experiments/my_model/__init__.py
from torchtitan.protocols.train_spec import TrainSpec

class MyModelArgs(TransformerModelArgs):
    custom_init_std: float = 0.01  # Custom parameter

class MyTransformer(Transformer):
    def init_weights(self, buffer_device=None):
        # Completely custom initialization
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.orthogonal_(param)  # Orthogonal init
            else:
                nn.init.zeros_(param)

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=MyTransformer,
        model_args={"custom": MyModelArgs()},
        # ... other spec fields
    )
```

### The ModelProtocol Requirement

All TorchTitan models must implement the `ModelProtocol` interface:

```python
# torchtitan/protocols/model.py
class ModelProtocol(Protocol):
    def __init__(self, model_args: BaseModelArgs) -> None: ...

    @abstractmethod
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights. REQUIRED for all models."""
        pass

    def forward(self, *args, **kwargs) -> Any: ...
```

**If your model doesn't implement `init_weights()`**, training will fail when
TorchTitan calls `model.init_weights()` after `to_empty()`.

### Common Initialization Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| No `init_weights()` method | Uninitialized weights (garbage values) | Implement `init_weights()` |
| Using PyTorch defaults | Kaiming init may not suit transformers | Use truncated normal |
| Forgetting new layers | Custom layers stay uninitialized | Add to parent's `init_weights()` |
| Wrong std for depth | All layers same scale | Use depth-scaled std for residual paths |
| Not calling `reset_parameters()` | Norm layers may have wrong values | Call for RMSNorm/LayerNorm |

---

## Activation Checkpointing

Activation checkpointing trades compute for memory by recomputing activations during
backward pass instead of storing them.

### Modes

```toml
[activation_checkpoint]
mode = "selective"          # "selective", "full", "memory_budget", "none"
selective_ac_option = "2"   # Every 2nd layer, or "op" for operation-level
```

| Mode | Description | Memory Savings | Compute Overhead |
|------|-------------|----------------|------------------|
| `none` | No checkpointing | None | None |
| `selective` | Checkpoint every Nth layer | Medium | Low |
| `full` | Checkpoint all layers | High | High |
| `memory_budget` | Compiler-guided optimal | Configurable | Configurable |

### Selective Options

```toml
# Checkpoint every 2nd layer
selective_ac_option = "2"

# Operation-level checkpointing (finer granularity)
selective_ac_option = "op"
```

### Memory Budget Mode

When using `memory_budget` mode, the compiler automatically determines optimal
checkpointing based on a memory/compute tradeoff:

```toml
[activation_checkpoint]
mode = "memory_budget"
memory_budget = 0.5         # 0.0 = max memory savings, 1.0 = no checkpointing
```

**Note:** Memory budget mode requires `torch.compile` to be enabled.

### Visualizing Memory Budget Pareto Frontier

To understand the memory vs. compute tradeoffs for your model, enable Pareto
visualization:

```toml
[activation_checkpoint]
mode = "memory_budget"
memory_budget = 0.5
visualize_memory_budget_pareto = true
```

**What it does:**
- Generates an SVG visualization showing runtime vs. activation memory tradeoffs
- Evaluates all memory budget values from 0.0 to 1.0 in increments of 0.05
- Helps you choose the optimal `memory_budget` value for your use case

**Output location:** `{job.dump_folder}/memory_budget_pareto/`

**How to use:**
1. Enable the visualization in your config
2. Run training (the visualization is generated during the first compiled forward pass)
3. Open the SVG file in the output directory to see the Pareto frontier
4. Use the chart to select a `memory_budget` value that balances your memory and compute needs:
   - Values closer to 0.0: Maximum memory savings, higher compute overhead
   - Values closer to 1.0: Minimal memory savings, lower compute overhead
5. Update your config with the chosen `memory_budget` value

For more details on the visualization format, see the
[PyTorch implementation example](https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015).

---

## Model Converters (Quantization)

Model converters transform the model before parallelization. Used for FP8 training
and other quantization schemes.

### FP8 Training (H100/H200)

```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true  # Save FSDP communication bandwidth
scaling_type_input = "delayed"
scaling_type_weight = "delayed"
```

**Requirements:**
- H100/H200 GPUs (SM89+)
- `torch.compile` enabled

### MXFP8 Training (B200)

```toml
[model]
converters = ["quantize.linear.mx"]

[quantize.linear.mx]
recipe_name = "mxfp8_cublas"           # cuBLAS kernels for B200
mxfp8_dim1_cast_kernel_choice = "cuda"
filter_fqns = ["output"]               # Skip output layer for stability
```

**Requirements:**
- B200/GB200 GPUs
- TorchAO from source for GB200

### How Converters Work

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Model Converter Timing                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Model Definition    Model Converters      Parallelization         │
│   (meta device)           apply            (TP, FSDP, PP)           │
│        │                    │                    │                  │
│        ▼                    ▼                    ▼                  │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐            │
│   │ Create  │   ───►   │ Swap    │   ───►   │ Shard   │            │
│   │ model   │          │ modules │          │ params  │            │
│   │ on meta │          │ (FP8,   │          │ (FSDP)  │            │
│   │ device  │          │  MX)    │          └─────────┘            │
│   └─────────┘          └─────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Converters run **before** parallelization because they may:
- Add new parameters (e.g., FP8 scale tensors)
- Change module structure (e.g., replace `nn.Linear` with `Float8Linear`)
- Require the full module structure for FSDP wrapping

---

## RoPE (Rotary Position Embeddings)

TorchTitan uses RoPE for position encoding with scaling support for long contexts.

### Base Configuration

```python
rope_theta: float = 500000  # Base frequency (Llama 3 default)
```

Higher `rope_theta` enables better long-context generalization.

### RoPE Scaling for Long Context

For context lengths beyond the original training length:

```python
@dataclass
class RoPEScalingArgs:
    scaling_factor: float = 8.0      # Overall scaling (higher = longer context)
    low_freq_factor: float = 1.0     # Low-frequency component scaling
    high_freq_factor: float = 4.0    # High-frequency component scaling
    original_max_position_embeddings: int = 8192  # Original context length
```

This implements the Llama 3.1 RoPE scaling scheme that interpolates position
embeddings for sequences longer than the original training context.

---

## Grouped Query Attention (GQA)

Most modern models use GQA for memory efficiency:

```python
n_heads: int = 32      # Query heads
n_kv_heads: int = 8    # Key/Value heads (fewer = more efficient)
```

**Head ratio:** `n_heads / n_kv_heads` determines memory savings
- Llama 3 8B: 32/8 = 4x KV cache reduction
- Llama 3 70B: 64/8 = 8x KV cache reduction

---

## FFN Hidden Dimension

The FFN hidden dimension is calculated with hardware-friendly rounding:

```python
# Base calculation
hidden_dim = 4 * dim
hidden_dim = int(2 * hidden_dim / 3)

# Apply multiplier if specified
if ffn_dim_multiplier:
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)

# Round to multiple_of for hardware efficiency
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
```

**Why `multiple_of` matters:**
- Aligns tensor dimensions for efficient GPU compute
- Larger models use larger multiples (1024, 2048, 4096)
- Prevents inefficient memory access patterns

---

## torch.compile Integration

TorchTitan leverages `torch.compile` for performance optimization.

### Configuration

```toml
[compile]
enable = true
components = ["model", "loss"]  # What to compile
```

### What Gets Compiled

| Component | Effect |
|-----------|--------|
| `model` | Compiles forward pass (Triton kernels, fusion) |
| `loss` | Compiles loss computation |

### When Compilation is Required

- **FP8 training**: Compile required for Float8 kernels
- **MXFP8 training**: Compile required for MX kernels
- **Flex attention**: Uses compiled Triton kernels
- **Varlen attention**: Uses compiled attention

### Compile Modes

TorchTitan uses specific compile options for different components:

```python
# Flex attention compilation
torch.compile(
    flex_attention,
    options={
        "max_autotune": True,
        "coordinate_descent_tuning": True,
        "triton.cudagraphs": False,
    },
)
```

---

## Mixed Precision Training

### Configuration

```toml
[training]
mixed_precision = "bfloat16"    # Compute precision
```

### Precision Modes

| Mode | Parameter Storage | Compute | Use Case |
|------|-------------------|---------|----------|
| `bfloat16` | BF16 | BF16 | Standard training |
| `float32` | FP32 | FP32 | Debugging, baseline |

With FP8/MXFP8 converters, matmul operations run in 8-bit while other ops stay
in the configured precision.

---

## Normalization

TorchTitan uses RMSNorm (Root Mean Square Layer Normalization):

```python
norm_eps: float = 1e-5  # Epsilon for numerical stability (Llama)
norm_eps: float = 1e-6  # Qwen3 uses smaller epsilon
```

RMSNorm is computationally cheaper than LayerNorm and works well for LLMs.

---

## Model Selection Guide

| Use Case | Model | Flavor | Key Settings |
|----------|-------|--------|--------------|
| Quick testing | `llama3` | `debugmodel` | Small, fast iteration |
| Document masking | `llama3` | `8B_flex` | `attn_type="flex"` |
| Long context + CP | `llama3` | `8B` | `context_parallel_degree > 1` |
| Production 8B | `llama3` | `8B` | Default settings |
| FP8 training | Any | Any | `converters=["quantize.linear.float8"]` |
| MXFP8 on B200 | Any | Any | `converters=["quantize.linear.mx"]` |

---

## Tips and Best Practices

### Memory Optimization

1. **Start with selective AC**: `mode="selective"`, `selective_ac_option="2"`
2. **Add FP8 for H100**: Reduces memory and improves throughput
3. **Use GQA models**: Llama 3 has 4-8x KV cache reduction
4. **Enable CPU offload**: For very large models, `enable_cpu_offload=true`

### Training Stability

1. **Use gradient clipping**: `max_norm=1.0` for most cases
2. **Enable depth init**: Default `depth_init=True` helps deep networks
3. **Lower LR for fine-tuning**: 1e-4 to 1e-5 vs 3e-4 for pretraining
4. **Skip output layer in FP8**: `filter_fqns=["output"]` for stability

### Performance

1. **Always compile**: `enable=true` for FP8, flex attention, varlen
2. **Match seq_len to parallelism**: Must be divisible by `tp * 2 * cp`
3. **Use appropriate multiple_of**: Matches model size (1024 for 8B, 4096 for 70B+)
4. **Tune batch size**: Maximize GPU utilization without OOM

### Debugging

1. **Use debugmodel**: Fast iteration with small model
2. **Disable compile**: Easier stack traces without compilation
3. **Set deterministic seeds**: `[debug]` section for reproducibility
4. **Check attention type**: SDPA for CP, flex/varlen for document masking
