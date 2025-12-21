# Parallelism Primer (DDP, FSDP, TP, PP, CP, EP)

This primer explains the parallelism options used in TorchTitan. It assumes you
know PyTorch, but not distributed LLM training.

## The core idea: split work across dimensions

TorchTitan composes multiple parallelism dimensions. Each dimension divides the
work in a different way:

- **Data Parallel (DP)**: replicate the model, split the batch.
- **Fully Sharded Data Parallel (FSDP)**: shard model parameters (and optimizer
  state/gradients) across DP ranks to reduce memory.
- **Hybrid Sharded Data Parallel (HSDP)**: combine DP replication and sharding.
- **Tensor Parallel (TP)**: shard large matrix multiplications by splitting
  tensor dimensions (e.g., shard columns/rows of linear layers).
- **Pipeline Parallel (PP)**: split the model into stages and pipeline
  microbatches through those stages.
- **Context Parallel (CP)**: shard the sequence dimension for attention to
  enable very long contexts.
- **Expert Parallel (EP/ETP)**: shard MoE experts across devices, optionally
  with different TP for experts.

TorchTitan builds a *device mesh* and maps these dimensions to mesh axes.

## Device mesh and dimension names

The `ParallelDims` class in `torchtitan/distributed/parallel_dims.py` creates
meshes with these logical names:

- `pp`: pipeline dimension
- `batch`: combined DP replicate + DP shard dimension (used by dataloading)
- `loss`: combined DP replicate + DP shard + CP dimension (used to reduce loss)
- `dp_replicate`: pure replication dimension (DDP/HSDP)
- `fsdp`: DP shard + CP (sharded parameters and reduce-scatter)
- `cp`: context parallel dimension
- `tp`: tensor parallel dimension
- `ep`: expert parallel dimension (MoE)
- `efsdp`: FSDP mesh used specifically for MoE experts
- `etp`: tensor parallel dimension for experts

Understanding these names makes it easier to read the code and interpret logs.

## How the degrees multiply

The world size must satisfy:

```
world_size = dp_replicate * dp_shard * cp * tp * pp
```

- `dp_shard` can be `-1`, which means "use leftover ranks" after multiplying the
  other degrees.
- If `dp_replicate > 1` and `dp_shard > 1`, you are using **HSDP**.
- If `dp_replicate = 1` and `dp_shard > 1`, you are using **FSDP**.
- If `dp_replicate > 1` and `dp_shard = 1`, you are using **DDP**.

## When to use each parallelism

- **FSDP/HSDP**: primary tool to reduce memory usage. Usually the first knob
  to turn for large models.
- **TP**: reduces compute per device for large matrix multiplies. Often paired
  with FSDP for large models.
- **PP**: useful when the model does not fit even with FSDP/TP. Adds pipeline
  bubbles and scheduling complexity.
- **CP**: allows very long context length by sharding sequence length. Useful
  for long-context models or inference with huge sequence lengths.
- **EP/ETP**: for MoE models only. Use EP to shard experts and ETP to adjust
  expert-specific tensor parallelism.

## HSDP: Combining DDP Replication and FSDP Sharding

When both `dp_replicate > 1` AND `dp_shard > 1`, you get **HSDP** (Hybrid
Sharded Data Parallel), which combines the benefits of both:

```
                     HSDP Architecture (dp_replicate=2, dp_shard=4, tp=2)
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                        dp_replicate = 2 (DDP-style)                          │
   │                                                                              │
   │          Replica Group 0                         Replica Group 1             │
   │   ┌───────────────────────────┐           ┌───────────────────────────┐      │
   │   │     dp_shard = 4          │           │     dp_shard = 4          │      │
   │   │  (FSDP shards params)     │           │  (FSDP shards params)     │      │
   │   │                           │           │                           │      │
   │   │  Shard0  Shard1  Shard2  Shard3       Shard0  Shard1  Shard2  Shard3     │
   │   │    │       │       │       │            │       │       │       │        │
   │   │   tp=2   tp=2    tp=2    tp=2          tp=2   tp=2    tp=2    tp=2       │
   │   │  ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐        ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐     │
   │   │  G0 G1   G2 G3   G4 G5   G6 G7        G8 G9  G10 G11 G12 G13 G14 G15    │
   │   └───────────────────────────┘           └───────────────────────────┘      │
   │                                                                              │
   └──────────────────────────────────────────────────────────────────────────────┘
```

**Communication pattern**:
- **Within each replica group (dp_shard)**: FSDP shards parameters, uses
  all-gather (forward) and reduce-scatter (backward)
- **Across replica groups (dp_replicate)**: DDP-style gradient averaging with
  all-reduce

**Why use HSDP?**
- **FSDP within nodes**: Saves memory by sharding parameters across local GPUs
  (fast NVLink communication)
- **DDP across nodes**: Reduces cross-node communication (only gradient
  averaging, not parameter gathering)

### What `dp_shard=-1` Means

The value `-1` means "auto-calculate from remaining GPUs after other
parallelisms":

```python
# From parallel_dims.py
if dp_shard < 0:
    self.dp_shard = world_size // (dp_replicate * cp * tp * pp)
```

**Example calculation**:
```
world_size = 16 GPUs
dp_replicate = 2  (2 replica groups)
tp = 2            (2-way tensor parallel)
pp = 1            (no pipeline parallel)
cp = 1            (no context parallel)

dp_shard = 16 // (2 * 1 * 2 * 1) = 4
```

### How Parallelism Dimensions Compose

The parallelism dimensions are orthogonal and compose multiplicatively. Each
GPU holds `1/(tp × dp_shard)` of the model within a replica:

```
                   Parameter Sharding Example
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   Original Weight W [4096 × 4096]                                  │
   │            │                                                       │
   │            ▼                                                       │
   │   ┌─────────────────────────────────────────────────────────┐     │
   │   │            TP Sharding (column-split by 2)              │     │
   │   │                                                         │     │
   │   │   W_tp0 [4096 × 2048]      W_tp1 [4096 × 2048]         │     │
   │   │         │                         │                     │     │
   │   │         ▼                         ▼                     │     │
   │   │   ┌─────────────────────────────────────────────────┐  │     │
   │   │   │      FSDP Sharding (row-split by 4)             │  │     │
   │   │   │                                                 │  │     │
   │   │   │  Shard0: W_tp0[0:1024,:]     W_tp1[0:1024,:]   │  │     │
   │   │   │          on G0               on G1             │  │     │
   │   │   │                                                 │  │     │
   │   │   │  Shard1: W_tp0[1024:2048,:]  W_tp1[1024:2048,:]│  │     │
   │   │   │          on G2               on G3             │  │     │
   │   │   │                                                 │  │     │
   │   │   │  Shard2: W_tp0[2048:3072,:]  W_tp1[2048:3072,:]│  │     │
   │   │   │          on G4               on G5             │  │     │
   │   │   │                                                 │  │     │
   │   │   │  Shard3: W_tp0[3072:4096,:]  W_tp1[3072:4096,:]│  │     │
   │   │   │          on G6               on G7             │  │     │
   │   │   └─────────────────────────────────────────────────┘  │     │
   │   └─────────────────────────────────────────────────────────┘     │
   │                                                                    │
   │   Each GPU holds: [1024 × 2048] = 1/8 of original weight          │
   │                   (1/4 from FSDP × 1/2 from TP)                   │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

**Communication during forward pass for one layer**:

1. **FSDP All-Gather** (within replica, across dp_shard dimension):
   - G0,G1 gather params from G2,G3,G4,G5,G6,G7 (and vice versa)
   - Now each GPU pair has full TP-sharded layer

2. **TP Compute + Communication** (within tp dimension):
   - G0 and G1 each compute partial result
   - All-reduce between G0 and G1 to combine

3. **FSDP Re-shard** (optional, if `reshard_after_forward=true`):
   - G0,G1 discard params they gathered, keep only their 1/4

**After backward**:
4. **DDP All-Reduce** (across dp_replicate dimension):
   - Average gradients between Replica 0 and Replica 1
   - G0 averages with G8, G1 with G9, etc.

| Dimension | What it shards | Communication | When |
|-----------|----------------|---------------|------|
| TP | Weight columns/rows | All-reduce per layer | During compute |
| FSDP | Weight rows (TP-sharded) | All-gather/reduce-scatter | Before/after compute |
| DDP | Nothing (full replica) | All-reduce gradients | After backward |

## Important constraints and gotchas

- **Sequence length divisibility**:
  - TP requires `seq_len` divisible by TP degree.
  - CP requires `seq_len` divisible by `2 * CP` (when load balancing).

- **Pipeline microbatching**:
  - `pipeline_parallel_microbatch_size` must divide `training.local_batch_size`.
  - Too few microbatches vs stages increases pipeline bubbles.

- **Mixed precision behavior**:
  - If FSDP/CP is enabled, mixed precision is handled inside FSDP.
  - If only DDP or single GPU is used, AMP (`torch.autocast`) is applied.

- **Loss parallel**:
  - When TP is enabled, the output is sharded. Loss parallel reduces the loss
    across those shards. Disable only when you know your loss is already global.
  - See [Loss Parallel](#loss-parallel-with-tensor-parallelism) below for details.

- **FP8 + TP**:
  - Tensorwise FP8 scaling can use FP8 all-gather with TP.
  - Rowwise FP8 uses higher-precision communication.

## Model Architecture Constraints for Parallelism

When using Tensor Parallelism (TP) or Context Parallelism (CP), certain model
architecture parameters must satisfy divisibility constraints. These constraints
arise from how the parallelism strategies shard weights and activations.

### Tensor Parallelism Constraints

TP shards weight matrices across GPUs. For this to work, the dimensions being
split must be evenly divisible by the TP degree.

#### 1. Attention Heads

```
Constraint: n_heads % tp_degree == 0
            n_kv_heads % tp_degree == 0
```

**Why**: TP applies `ColwiseParallel` to the Q, K, V projection layers (wq, wk,
wv), splitting them column-wise. Each TP rank receives a subset of attention
heads:

```
                    Attention Head Sharding (tp=4)

Original wq weight: [hidden_dim, n_heads × head_dim]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GPU 0: wq[:, 0:n_heads/4 × head_dim]      ← heads 0 to n_heads/4-1  │
│ GPU 1: wq[:, n_heads/4:n_heads/2 × head_dim]  ← heads n_heads/4...  │
│ GPU 2: wq[:, n_heads/2:3*n_heads/4 × head_dim]                      │
│ GPU 3: wq[:, 3*n_heads/4:n_heads × head_dim]                        │
└─────────────────────────────────────────────────────────────────────┘
```

If `n_heads` or `n_kv_heads` is not divisible by `tp_degree`, the sharding
would produce unequal chunks, causing dimension mismatches.

**Example**: Llama 3 8B has `n_heads=32` and `n_kv_heads=8`
- TP=2: ✓ (32/2=16 heads, 8/2=4 KV heads per GPU)
- TP=4: ✓ (32/4=8 heads, 8/4=2 KV heads per GPU)
- TP=8: ✓ (32/8=4 heads, 8/8=1 KV head per GPU)
- TP=16: ✗ (8/16=0.5 KV heads - not divisible!)

#### 2. FFN Hidden Dimension

```
Constraint: ffn_hidden_dim % tp_degree == 0
```

**Why**: TP applies `ColwiseParallel` to FFN up-projections (w1, w3) and
`RowwiseParallel` to the down-projection (w2):

```
FFN Sharding (tp=2):

w1: [hidden, ffn_dim]  →  GPU0: [hidden, ffn_dim/2]
                          GPU1: [hidden, ffn_dim/2]

w2: [ffn_dim, hidden]  →  GPU0: [ffn_dim/2, hidden]
                          GPU1: [ffn_dim/2, hidden]
```

The FFN hidden dimension (often 4× or 8/3× the model dimension) must be
divisible by `tp_degree`.

**Example**: With `hidden_dim=4096` and standard 8/3× multiplier:
- `ffn_dim = 4096 × 8/3 ≈ 10923` → rounded to `multiple_of` (e.g., 14336)
- TP=2: ✓ (14336/2=7168)
- TP=8: ✓ (14336/8=1792)

#### 3. Vocabulary Size (Embedding and Output Layers)

```
Constraint: vocab_size % tp_degree == 0
```

**Why**: Both the embedding layer and output projection layer involve the
vocabulary dimension, and both are sharded across TP ranks.

**Embedding layer** (`tok_embeddings`): Uses `RowwiseParallel`, which shards
the embedding weight matrix along the vocabulary (row) dimension:

```
Embedding Weight Sharding (tp=4, vocab=128K):

nn.Embedding weight: [vocab_size, dim]  →  [128K, 4096]
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GPU 0: weight[0:32K, :]       ← embeddings for vocab tokens 0-32K    │
│ GPU 1: weight[32K:64K, :]     ← embeddings for vocab tokens 32K-64K  │
│ GPU 2: weight[64K:96K, :]     ← embeddings for vocab tokens 64K-96K  │
│ GPU 3: weight[96K:128K, :]    ← embeddings for vocab tokens 96K-128K │
└──────────────────────────────────────────────────────────────────────┘

Input token IDs are Replicated across TP ranks.
Each rank looks up embeddings only for its vocab shard.
Output is Shard(1) - sharded on sequence dimension for Sequence Parallel.
```

**Output layer** (`output`): Uses `ColwiseParallel`, which shards the
projection weight along the vocabulary (column) dimension:

```
Output Layer Sharding (tp=4, vocab=128K):

output weight: [dim, vocab_size]  →  [4096, 128K]
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GPU 0: weight[:, 0:32K]       ← logits for vocab tokens 0-32K        │
│ GPU 1: weight[:, 32K:64K]     ← logits for vocab tokens 32K-64K      │
│ GPU 2: weight[:, 64K:96K]     ← logits for vocab tokens 64K-96K      │
│ GPU 3: weight[:, 96K:128K]    ← logits for vocab tokens 96K-128K     │
└──────────────────────────────────────────────────────────────────────┘
```

Most LLM vocabularies (32K, 128K, 256K) are powers of 2, making them divisible
by common TP degrees (2, 4, 8).

#### 4. Loss Parallel (Requires TP)

Loss Parallel is **not a separate parallelism dimension**—it is an optimization
that works with Tensor Parallelism. Key points:

```python
# From train.py - Loss Parallel requires TP to be enabled
loss_parallel_enabled = (
    parallel_dims.tp_enabled
    and not job_config.parallelism.disable_loss_parallel
)
```

**Relationship to TP**:
- Loss Parallel uses the **same mesh and degree as TP**
- It is enabled automatically when TP is enabled (unless explicitly disabled)
- The output logits are already sharded across TP ranks (from ColwiseParallel)
- Loss Parallel computes cross-entropy on each TP rank's vocab shard, then
  reduces the loss across ranks

```
Loss Parallel Flow (tp=4):

                    Logits: [batch, seq, 32K] per GPU (vocab sharded)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            GPU 0: CE loss    GPU 1: CE loss    GPU 2: CE loss    GPU 3: CE loss
            (vocab 0-32K)     (vocab 32K-64K)   (vocab 64K-96K)   (vocab 96K-128K)
                    │               │               │               │
                    └───────────────┴───────────────┴───────────────┘
                                    │
                                    ▼
                            All-reduce loss
                                    │
                                    ▼
                            Final scalar loss
```

**Memory benefit**: Without loss parallel, each GPU would need to gather the
full [batch, seq, vocab_size] logits tensor before computing loss. With
vocab=128K and seq=4K in BF16, that's 1GB per sample—loss parallel avoids this.

### Context Parallelism Constraints

CP splits the sequence across GPUs using ring attention.

#### Sequence Length

```
Constraint: seq_len % (tp_degree × cp_degree × 2) == 0
```

**Why**: The factor of 2 comes from load balancing in ring attention. CP
processes causal attention by splitting the sequence into chunks, and the ring
rotation pattern requires balanced chunk sizes:

```
Ring Attention Load Balancing (cp=4, seq=8192):

Each chunk: 8192 / (4 × 2) = 1024 tokens

The ×2 factor accounts for:
- Causal masking creates triangular attention patterns
- Without balancing, early chunks compute more attention than late chunks
- The factor of 2 ensures work is evenly distributed across ring steps
```

**Example calculations**:
- seq_len=4096, tp=1, cp=4: 4096 % (1×4×2) = 4096 % 8 = 0 ✓
- seq_len=4096, tp=2, cp=2: 4096 % (2×2×2) = 4096 % 8 = 0 ✓
- seq_len=32768, tp=1, cp=8: 32768 % (1×8×2) = 32768 % 16 = 0 ✓
- seq_len=1000, tp=1, cp=4: 1000 % 8 = 0 ✓
- seq_len=1000, tp=2, cp=4: 1000 % 16 = 8 ✗

### Summary Table

| Parameter | Constraint | Applies To | Reason |
|-----------|------------|------------|--------|
| `n_heads` | `% tp == 0` | TP | Query heads split across TP ranks |
| `n_kv_heads` | `% tp == 0` | TP | KV heads split across TP ranks |
| `ffn_hidden_dim` | `% tp == 0` | TP | FFN layers split column/row-wise |
| `vocab_size` | `% tp == 0` | TP | Embedding rows and output columns sharded |
| `seq_len` | `% (tp × cp × 2) == 0` | CP | Ring attention load balancing |

### Practical Guidance

1. **Choose TP degree based on KV heads**: The `n_kv_heads` is often the most
   restrictive constraint. Llama models use 8 KV heads, limiting TP to 1, 2, 4,
   or 8.

2. **Verify before training**: Check divisibility before launching:
   ```python
   assert model_args.n_heads % tp_degree == 0, \
       f"n_heads ({model_args.n_heads}) must be divisible by tp ({tp_degree})"
   assert model_args.n_kv_heads % tp_degree == 0, \
       f"n_kv_heads ({model_args.n_kv_heads}) must be divisible by tp ({tp_degree})"
   ```

3. **Common valid configurations**:
   | Model | n_heads | n_kv_heads | Valid TP degrees |
   |-------|---------|------------|------------------|
   | Llama 3 8B | 32 | 8 | 1, 2, 4, 8 |
   | Llama 3 70B | 64 | 8 | 1, 2, 4, 8 |
   | Llama 3 405B | 128 | 8 | 1, 2, 4, 8 |
   | Qwen3 1.7B | 16 | 2 | 1, 2 |
   | Qwen3 32B | 64 | 8 | 1, 2, 4, 8 |

4. **Adjust seq_len for CP**: Round sequence length to satisfy the constraint:
   ```python
   divisor = tp_degree * cp_degree * 2
   seq_len = (desired_seq_len // divisor) * divisor
   ```

## Typical configurations

- **Single GPU debug**:
  - dp_replicate=1, dp_shard=1, tp=1, pp=1, cp=1

- **8-GPU data parallel**:
  - dp_replicate=8, dp_shard=1, tp=1, pp=1

- **8-GPU FSDP**:
  - dp_replicate=1, dp_shard=8, tp=1, pp=1

- **8-GPU HSDP (2x4)**:
  - dp_replicate=2, dp_shard=4, tp=1, pp=1

- **16-GPU TP+FSDP (4x4)**:
  - dp_replicate=1, dp_shard=4, tp=4, pp=1

- **Pipeline parallel (4 stages)**:
  - pp=4, dp_shard=1 or 4 (paired with FSDP), tp=1

## Where these settings live

All parallelism settings live under the `[parallelism]` section in TOML and the
`JobConfig.parallelism` dataclass. The full option list and defaults are in
`docs/config_reference.md`.

---

## Loss Parallel with Tensor Parallelism

### What Loss Parallel Is

Loss Parallel shards the vocabulary dimension of the final linear layer (the
"output head") and cross-entropy loss computation across TP ranks.

### Why It Exists

For LLMs, vocabulary size is huge (32K-256K tokens). The output logits tensor
has shape `[batch, seq, vocab_size]`. This is memory-intensive:

```
4K seq × 128K vocab × 2 bytes (BF16) = 1GB per sample!
```

### How It Works

```
Without Loss Parallel:
┌──────────────────────────────────────────────────────────────┐
│ GPU 0: Full logits [batch, seq, 128K vocab]      ← 1GB      │
│ GPU 1: Full logits [batch, seq, 128K vocab]      ← 1GB      │
│        Cross-entropy computed on full vocab                  │
└──────────────────────────────────────────────────────────────┘

With Loss Parallel:
┌──────────────────────────────────────────────────────────────┐
│ GPU 0: Partial logits [batch, seq, 64K vocab]    ← 0.5GB    │
│ GPU 1: Partial logits [batch, seq, 64K vocab]    ← 0.5GB    │
│        Cross-entropy computed in parallel                    │
│        Final loss reduced across ranks                       │
└──────────────────────────────────────────────────────────────┘
```

### The `disable_loss_parallel` Config Option

Loss parallel **requires** TP to be enabled (that's where the sharding
happens). The config option `disable_loss_parallel` allows you to turn it OFF
when TP is on:

```python
loss_parallel_enabled = (
    parallel_dims.tp_enabled
    and not job_config.parallelism.disable_loss_parallel
)
```

**When you might disable it**:
1. **Debugging**: To compare results without loss parallel
2. **Compatibility**: Some custom loss functions may not support loss parallel
3. **Small vocab**: If vocabulary is small, the memory benefit is minimal

---

## Async Tensor Parallelism

### Standard TP Communication Pattern

In normal TP, each layer requires synchronous communication:

```
Standard TP (Synchronous):

GPU 0: ┌─────────┐     ┌──────────┐     ┌─────────┐
       │ Compute │ ──► │ AllReduce│ ──► │ Compute │ ──► ...
       └─────────┘     └──────────┘     └─────────┘
                            ↑
                       SYNC POINT (all GPUs wait)
                            ↑
GPU 1: ┌─────────┐     ┌──────────┐     ┌─────────┐
       │ Compute │ ──► │ AllReduce│ ──► │ Compute │ ──► ...
       └─────────┘     └──────────┘     └─────────┘
```

**The problem**: GPUs sit idle during communication, even if they have
independent work to do.

### What Async TP Does

Async TP overlaps communication with computation using two techniques:

#### 1. Symmetric Memory

Instead of using NCCL's traditional collective operations, symmetric memory
allows GPUs to directly read/write each other's memory:

```
Traditional NCCL AllReduce:
- GPU 0 → Buffer → NCCL → Buffer → GPU 1
- Requires synchronization barriers
- Higher latency for small messages

Symmetric Memory:
- GPU 0 ←→ GPU 1 direct memory access
- No explicit barriers needed
- Lower latency, better for TP's frequent small collectives
```

#### 2. Micro-Pipelining (via torch.compile)

The compiler reorders operations to overlap communication with compute:

```
Without Micro-Pipelining:
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Compute1│→│  Comm  │→│Compute2│→│  Comm  │→ ...
└────────┘ └────────┘ └────────┘ └────────┘

With Micro-Pipelining (Async TP):
┌────────┐ ┌────────┐ ┌────────┐
│Compute1│→│Compute2│→│Compute3│→ ...
└────────┘ └────────┘ └────────┘
      ↓         ↓
   ┌──────┐ ┌──────┐
   │ Comm │ │ Comm │  (runs in parallel with compute)
   └──────┘ └──────┘
```

### How to Enable Async TP

```toml
[parallelism]
tensor_parallel_degree = 2
enable_async_tensor_parallel = true

[compile]
enable = true
components = ["model"]  # REQUIRED for async TP
```

**Requirements**:
1. `torch.compile` must be enabled for the model
2. Hardware with NVLink or similar high-bandwidth interconnect
3. SM89+ (H100) recommended for best performance

### Performance Impact

- Typically 5-15% throughput improvement for TP-heavy workloads
- Benefit increases with more TP communication (larger TP degree, smaller
  batch sizes)
- Less benefit if compute-bound (large batch sizes)

---

## Tensor Parallel Sharding Patterns

TorchTitan uses **both column-wise and row-wise** sharding for TP, depending on
the layer type. The pattern follows standard Megatron-LM conventions.

### Column-Wise Parallel (ColwiseParallel)

Weight matrix split along columns. Each GPU gets `[d_in, d_out/N]` slice:

```
Column-Wise Sharding (e.g., wq, wk, wv, w1, w3):

Original W [4096, 4096]
     │
     ▼
┌─────────────────────────────────────────────┐
│ GPU 0: W[:, 0:2048]     [4096, 2048]        │
│ GPU 1: W[:, 2048:4096]  [4096, 2048]        │
└─────────────────────────────────────────────┘

Communication: All-gather input before computation
```

**Used for**: Expansion layers (Q, K, V projections, FFN up-projections w1/w3)

### Row-Wise Parallel (RowwiseParallel)

Weight matrix split along rows. Each GPU gets `[d_in/N, d_out]` slice:

```
Row-Wise Sharding (e.g., wo, w2):

Original W [4096, 4096]
     │
     ▼
┌─────────────────────────────────────────────┐
│ GPU 0: W[0:2048, :]     [2048, 4096]        │
│ GPU 1: W[2048:4096, :]  [2048, 4096]        │
└─────────────────────────────────────────────┘

Communication: Reduce-scatter outputs after computation
```

**Used for**: Contraction layers (attention output wo, FFN down-projection w2)

### How Attention Layers Are Sharded

In TorchTitan, **attention layers use TP** (not FSDP) for the actual weight
sharding. FSDP then shards the already-TP-sharded parameters further.

```
Llama Attention TP Plan (from parallelize.py):

┌────────────────────────────────────────────────────────────────────────┐
│                         Attention Block                                │
│                                                                        │
│   Input: [batch, seq, hidden]  (Replicate or Shard(seq))              │
│                │                                                       │
│                ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │  wq, wk, wv: ColwiseParallel                                    │ │
│   │  - Each GPU has 1/N of attention heads                          │ │
│   │  - wq: [hidden, n_heads/N * head_dim]                           │ │
│   │  - wk: [hidden, n_kv_heads/N * head_dim]                        │ │
│   │  - wv: [hidden, n_kv_heads/N * head_dim]                        │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                │                                                       │
│                ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │  Attention Computation (local to each GPU)                      │ │
│   │  - Each GPU computes attention for its subset of heads          │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                │                                                       │
│                ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │  wo: RowwiseParallel                                            │ │
│   │  - Combines partial outputs from all heads                      │ │
│   │  - Reduce-scatter to get final output                           │ │
│   │  - Output layout: Shard(1) for sequence parallel                │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                │                                                       │
│                ▼                                                       │
│   Output: [batch, seq, hidden]  (Shard(seq) for memory efficiency)    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### FFN Sharding Pattern

```
FFN Block TP Plan:

Input: [batch, seq, hidden]
       │
       ├──► w1: ColwiseParallel ──► [batch, seq, ffn_dim/N]
       │                                    │
       │                                    ├──► SiLU activation
       │                                    │
       └──► w3: ColwiseParallel ──► [batch, seq, ffn_dim/N]
                                            │
                                            ▼
                                      Element-wise multiply
                                            │
                                            ▼
                                   w2: RowwiseParallel
                                            │
                                            ▼
                              Output: [batch, seq, hidden]
```

### Sequence Parallel Integration

Between TP operations, activations are kept **sharded on the sequence
dimension** (Shard(1)) to save memory. This is called Sequence Parallel:

```python
# From parallelize.py - the TP plan for each layer
layer_plan = {
    "attention_norm": SequenceParallel(),           # Keep seq sharded
    "attention": PrepareModuleInput(Shard(1) → Replicate()),  # Gather for compute
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),  # Back to seq sharded
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(Shard(1) → Replicate()),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
}
```

---

## Dataset Sharding in Multi-Dimensional Parallelism

### Key Principle: TP Ranks Share the Same Data

In TorchTitan, **only the data parallel dimensions** (dp_replicate × dp_shard)
affect dataset sharding. TP ranks within the same DP group see **identical**
data batches.

```
Dataset Sharding with dp=4, tp=2 (8 GPUs total):

┌─────────────────────────────────────────────────────────────────────────┐
│                          Dataset                                        │
│   [Sample 0, Sample 1, Sample 2, Sample 3, Sample 4, Sample 5, ...]    │
│        │           │           │           │                            │
│        ▼           ▼           ▼           ▼                            │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│   │DP Rank 0│ │DP Rank 1│ │DP Rank 2│ │DP Rank 3│                       │
│   │ G0, G1  │ │ G2, G3  │ │ G4, G5  │ │ G6, G7  │   ◄── TP pairs       │
│   │ (tp=2)  │ │ (tp=2)  │ │ (tp=2)  │ │ (tp=2)  │                       │
│   │         │ │         │ │         │ │         │                       │
│   │ SAME    │ │ SAME    │ │ SAME    │ │ SAME    │   ◄── G0,G1 see      │
│   │ batch!  │ │ batch!  │ │ batch!  │ │ batch!  │       same samples   │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘                       │
│                                                                         │
│   batch_degree = dp_replicate * dp_shard = 4                           │
│   Each DP rank gets 1/4 of the dataset                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Batch Mesh

TorchTitan creates a "batch" mesh dimension that combines all data-parallel
ranks:

```python
# From parallel_dims.py
batch = dp_replicate * dp_shard  # Total DP degree

# Dataloader uses batch mesh for sharding
batch_mesh = parallel_dims.get_mesh("batch")
dp_world_size = batch_mesh.size()    # = dp_replicate * dp_shard
dp_rank = batch_mesh.get_local_rank()  # This rank's position in DP

# Dataset is split by dp_rank
dataset = split_dataset_by_node(ds, dp_rank, dp_world_size)
```

### Why TP Ranks Share Data

TP shards **weights**, not data. Each TP rank:
- Receives the **same input batch**
- Computes on **different weight shards**
- Produces **partial outputs** that are combined via all-reduce/reduce-scatter

This is fundamentally different from data parallelism where each rank
processes different data with replicated weights.

### 3D Parallelism Data Flow Example

With `dp_shard=2, tp=2, pp=2` on 8 GPUs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        3D Parallelism Data Flow                        │
│                                                                         │
│   Dataset: [batch 0, batch 1, batch 2, ...]                            │
│                │           │                                            │
│           DP Rank 0    DP Rank 1                                       │
│                │           │                                            │
│                ▼           ▼                                            │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    Pipeline Stage 0                             │   │
│   │   ┌─────────────────────┐   ┌─────────────────────┐            │   │
│   │   │ G0, G1 (TP pair)    │   │ G2, G3 (TP pair)    │            │   │
│   │   │ Same batch 0        │   │ Same batch 1        │            │   │
│   │   │ Different weights   │   │ Different weights   │            │   │
│   │   └─────────────────────┘   └─────────────────────┘            │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                │                         │                              │
│          activations               activations                          │
│                │                         │                              │
│                ▼                         ▼                              │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    Pipeline Stage 1                             │   │
│   │   ┌─────────────────────┐   ┌─────────────────────┐            │   │
│   │   │ G4, G5 (TP pair)    │   │ G6, G7 (TP pair)    │            │   │
│   │   │ Receives from G0,G1 │   │ Receives from G2,G3 │            │   │
│   │   └─────────────────────┘   └─────────────────────┘            │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Context Parallelism (CP)

Context Parallelism shards the **sequence dimension** across GPUs to enable
training with very long contexts that wouldn't fit on a single device.

### The Problem: Attention Memory Scales Quadratically

Standard self-attention computes attention scores between all pairs of tokens:

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Memory for attention scores: O(seq_len²)
```

For a 32K sequence with batch=1 in BF16:
- Attention scores: 32K × 32K × 2 bytes = **2 GB per layer per head**
- With 32 heads: **64 GB just for attention scores**

This quickly exceeds GPU memory. Context Parallelism solves this by splitting
the sequence across GPUs, so each GPU only computes a portion of the attention.

### The Key Insight: Ring Attention

The insight behind ring attention is that attention can be computed
**incrementally**. Instead of computing the full attention matrix at once,
we can:

1. Compute attention for a subset of K,V (producing partial scores)
2. Combine partial results using the **online softmax** algorithm
3. Repeat until all K,V have been processed

This allows each GPU to hold only `seq_len/cp` tokens while still computing
correct attention over the full sequence.

### How Ring Attention Works

Consider CP=4 with seq_len=8192 (2048 tokens per GPU):

```
Sequence Distribution:

┌─────────────────────────────────────────────────────────────────────────┐
│ Original: [token_0, token_1, ..., token_8191]                           │
│                                                                         │
│ GPU 0: Q₀, K₀, V₀  (tokens 0-2047)                                     │
│ GPU 1: Q₁, K₁, V₁  (tokens 2048-4095)                                  │
│ GPU 2: Q₂, K₂, V₂  (tokens 4096-6143)                                  │
│ GPU 3: Q₃, K₃, V₃  (tokens 6144-8191)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

Each GPU needs to compute attention between its queries (Qᵢ) and ALL keys/values
(K₀...K₃, V₀...V₃). Ring attention achieves this through rotation:

```
Ring Attention Steps (cp=4):

Step 0: Local computation
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ GPU0 │    │ GPU1 │    │ GPU2 │    │ GPU3 │
│Q₀×K₀ │    │Q₁×K₁ │    │Q₂×K₂ │    │Q₃×K₃ │
└──────┘    └──────┘    └──────┘    └──────┘
   Each GPU computes attention with its local KV

Step 1: Rotate KV, compute with received KV
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ GPU0 │◄───│ GPU1 │◄───│ GPU2 │◄───│ GPU3 │◄──┐
│Q₀×K₃ │    │Q₁×K₀ │    │Q₂×K₁ │    │Q₃×K₂ │   │
└──────┘    └──────┘    └──────┘    └──────┘   │
   │                                           │
   └───────────────────────────────────────────┘
   KV shards rotate around the ring

Step 2: Rotate again
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ GPU0 │    │ GPU1 │    │ GPU2 │    │ GPU3 │
│Q₀×K₂ │    │Q₁×K₃ │    │Q₂×K₀ │    │Q₃×K₁ │
└──────┘    └──────┘    └──────┘    └──────┘

Step 3: Final rotation
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ GPU0 │    │ GPU1 │    │ GPU2 │    │ GPU3 │
│Q₀×K₁ │    │Q₁×K₂ │    │Q₂×K₃ │    │Q₃×K₀ │
└──────┘    └──────┘    └──────┘    └──────┘

After 4 steps, each GPU has computed Q×K for all K shards.
Results are combined using online softmax.
```

### Online Softmax: Combining Partial Results

The magic of ring attention is that we don't need to store all attention scores.
We use **online softmax** to incrementally update the output:

```
Online Softmax Algorithm:

For each incoming KV shard:
1. Compute local attention scores: scores = Qᵢ × Kⱼᵀ / √d
2. Find local max: m_new = max(m_old, max(scores))
3. Rescale previous output: out = out × exp(m_old - m_new)
4. Add new contribution: out += softmax(scores - m_new) × Vⱼ
5. Update normalization: l = l × exp(m_old - m_new) + sum(exp(scores - m_new))

Final: out = out / l
```

This produces mathematically identical results to standard attention while
using O(seq_len/cp) memory per GPU instead of O(seq_len²).

### Causal Masking and Load Balancing

For causal (autoregressive) attention, token i can only attend to tokens 0..i.
This creates a triangular attention pattern:

```
Causal Attention Matrix (seq=8, cp=2):

        K₀ (GPU 0)    K₁ (GPU 1)
       [0,1,2,3]      [4,5,6,7]
      ┌─────────────┬─────────────┐
Q₀  0 │  ■          │             │
    1 │  ■ ■        │             │
    2 │  ■ ■ ■      │             │
    3 │  ■ ■ ■ ■    │             │
      ├─────────────┼─────────────┤
Q₁  4 │  ■ ■ ■ ■    │  ■          │
    5 │  ■ ■ ■ ■    │  ■ ■        │
    6 │  ■ ■ ■ ■    │  ■ ■ ■      │
    7 │  ■ ■ ■ ■    │  ■ ■ ■ ■    │
      └─────────────┴─────────────┘

■ = valid attention (not masked)
```

**The Load Imbalance Problem**:
- GPU 0 (Q₀): Only attends to K₀ (triangular, ~50% of block)
- GPU 1 (Q₁): Attends to all of K₀ + triangular K₁ (~150% of one block)

GPU 1 does 3× more work than GPU 0!

**The Solution: Zigzag Splitting**

To balance load, CP uses a zigzag pattern that interleaves tokens:

```
Zigzag Token Assignment (seq=8, cp=2):

Instead of: GPU 0 = [0,1,2,3], GPU 1 = [4,5,6,7]

Zigzag:     GPU 0 = [0,2,4,6], GPU 1 = [1,3,5,7]   (even/odd)

Or more generally, split into 2×cp chunks and alternate:

seq=16, cp=2:
Chunk 0: [0,1,2,3]   → GPU 0
Chunk 1: [4,5,6,7]   → GPU 1
Chunk 2: [8,9,10,11] → GPU 1  (reversed assignment)
Chunk 3: [12,13,14,15] → GPU 0
```

This is why the constraint is `seq_len % (cp × 2) == 0` — the factor of 2
comes from this load-balancing zigzag pattern.

### CP Configuration Options

```toml
[parallelism]
context_parallel_degree = 4
context_parallel_rotate_method = "allgather"  # or "alltoall"
```

#### Rotation Methods

**`allgather` (default)**:
- After the first local attention, all-gather all KV shards
- Each GPU then has full KV and computes remaining attention locally
- **Pros**: Simple, fewer communication rounds
- **Cons**: Higher peak memory (holds all KV briefly)

```
AllGather Method:

Step 1: Local attention (Q₀×K₀ on GPU0, etc.)
Step 2: All-gather KV → each GPU has [K₀,K₁,K₂,K₃]
Step 3: Each GPU computes remaining attention locally
```

**`alltoall`**:
- KV shards are shuffled via all-to-all communication each step
- True ring rotation pattern
- **Pros**: Lower peak memory, better for very high CP degrees
- **Cons**: More communication rounds, higher latency

```
AllToAll Method:

Repeat cp times:
  1. Compute attention with current KV
  2. All-to-all shuffle: send KV to next rank, receive from previous
```

**When to use which**:
- `allgather`: CP ≤ 8, good intra-node bandwidth (NVLink)
- `alltoall`: CP > 8, or memory-constrained scenarios

### Memory and Communication Analysis

**Memory per GPU**:

| Component | Without CP | With CP (degree=4) |
|-----------|-----------|-------------------|
| Q, K, V tensors | O(seq × d) | O(seq/4 × d) |
| Attention scores | O(seq²) | O(seq²/16) * |
| Activations | O(seq × d) | O(seq/4 × d) |

\* With FlashAttention, attention scores are not materialized, but the
computational work is still reduced by cp² factor per GPU.

**Communication volume**:
- Each rotation step: 2 × (seq/cp) × n_kv_heads × head_dim × dtype_size
- Total rotations: cp - 1 (or 1 for allgather)
- Communication is overlapped with computation when possible

**Example for 32K context, cp=4, hidden=4096, 8 KV heads, BF16**:
- Per rotation: 2 × 8K × 8 × 128 × 2 bytes = 32 MB
- Total (alltoall): 3 × 32 MB = 96 MB per layer
- Total (allgather): 4 × 32 MB = 128 MB (one-time)

### CP + TP + FSDP Mesh Topology

When combining CP with other parallelisms, the mesh is organized hierarchically.
With `dp_shard=2, tp=2, cp=2` on 8 GPUs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CP + TP + FSDP Mesh (8 GPUs)                        │
│                                                                         │
│   Mesh dimensions: (dp_shard=2) × (cp=2) × (tp=2) = 8                  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    FSDP Shard 0                                  │  │
│   │   ┌─────────────────────────┐   ┌─────────────────────────┐     │  │
│   │   │      CP Rank 0          │   │      CP Rank 1          │     │  │
│   │   │   ┌───────┬───────┐     │   │   ┌───────┬───────┐     │     │  │
│   │   │   │ G0    │ G1    │     │   │   │ G2    │ G3    │     │     │  │
│   │   │   │ TP=0  │ TP=1  │     │   │   │ TP=0  │ TP=1  │     │     │  │
│   │   │   │seq 0-L│seq 0-L│     │   │   │seq L-N│seq L-N│     │     │  │
│   │   │   └───────┴───────┘     │   │   └───────┴───────┘     │     │  │
│   │   └─────────────────────────┘   └─────────────────────────┘     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    FSDP Shard 1                                  │  │
│   │   ┌─────────────────────────┐   ┌─────────────────────────┐     │  │
│   │   │      CP Rank 0          │   │      CP Rank 1          │     │  │
│   │   │   ┌───────┬───────┐     │   │   ┌───────┬───────┐     │     │  │
│   │   │   │ G4    │ G5    │     │   │   │ G6    │ G7    │     │     │  │
│   │   │   │ TP=0  │ TP=1  │     │   │   │ TP=0  │ TP=1  │     │     │  │
│   │   │   │seq 0-L│seq 0-L│     │   │   │seq L-N│seq L-N│     │     │  │
│   │   │   └───────┴───────┘     │   │   └───────┴───────┘     │     │  │
│   │   └─────────────────────────┘   └─────────────────────────┘     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Key relationships:                                                    │
│   - TP pairs (G0,G1), (G2,G3), etc. share same data AND same seq chunk │
│   - CP pairs (G0,G2), (G1,G3), etc. share same data, different seq     │
│   - FSDP groups (G0-G3), (G4-G7) have different data entirely          │
│   - fsdp mesh = dp_shard × cp = 2 × 2 = 4                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Communication patterns**:
- **TP**: All-reduce within (G0,G1), (G2,G3), etc. — every layer
- **CP**: Ring rotation within (G0,G2), (G1,G3), etc. — during attention only
- **FSDP**: All-gather/reduce-scatter within (G0-G3), (G4-G7) — before/after layers

### Sequence Length Constraint Explained

```
Constraint: seq_len % (tp_degree × cp_degree × 2) == 0
```

This constraint comes from three requirements:

1. **CP chunking**: seq_len must divide evenly by cp_degree
2. **Load balancing**: The zigzag pattern requires 2× chunks (factor of 2)
3. **TP integration**: When TP is enabled, sequence is also sharded for
   Sequence Parallel, requiring divisibility by tp_degree

```python
# From TorchTitan's parallel_dims.py
seq_len_divisor = tp * (cp * 2)
assert seq_len % seq_len_divisor == 0, \
    f"seq_len ({seq_len}) must be divisible by {seq_len_divisor}"
```

**Example calculations**:
- seq_len=4096, tp=1, cp=4: 4096 % (1×4×2) = 4096 % 8 = 0 ✓
- seq_len=32768, tp=2, cp=4: 32768 % (2×4×2) = 32768 % 16 = 0 ✓
- seq_len=10000, tp=1, cp=4: 10000 % 8 = 0 ✓
- seq_len=10000, tp=2, cp=4: 10000 % 16 = 0 ✗ (need 10000→9984 or 10000→10000)

### When to Use Context Parallelism

**Use CP when**:
- Sequence length exceeds single-GPU memory (typically >8K-16K)
- Training long-context models (32K, 64K, 128K+)
- Fine-tuning for extended context capability

**Don't use CP when**:
- Sequence length fits in memory (adds unnecessary communication)
- CP degree would exceed practical limits (communication overhead)
- Model doesn't support SDPA attention (CP requires it)

**Typical configurations**:

| Sequence Length | Recommended CP | Chunk Size per GPU |
|-----------------|----------------|-------------------|
| 4K | 1 (disabled) | 4K |
| 8K-16K | 2 | 4K-8K |
| 32K | 4 | 8K |
| 64K | 8 | 8K |
| 128K | 8-16 | 8K-16K |

**Rule of thumb**: Target 4K-8K tokens per GPU per CP chunk for optimal
efficiency. Going below 2K increases communication overhead.

---

## Parallelism Selection Guide for Llama3-like Models on B200 GPUs

This section provides practical guidance on selecting parallelism strategies
based on model size for Llama3-like transformer architectures. All calculations
assume NVIDIA B200 GPUs (192 GB HBM3e) with 8 GPUs per node, 8K context length,
and BF16 mixed precision training.

### Memory Requirements for Training

Training a model requires memory for multiple components:

```
Total Training Memory = Parameters + Gradients + Optimizer States + Activations

Components (BF16 mixed precision with AdamW):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Component          │ Bytes per Parameter │ Notes                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Parameters (BF16)  │ 2 bytes             │ Model weights                   │
│ Gradients (BF16)   │ 2 bytes             │ Same size as params             │
│ Optimizer States   │ 8 bytes             │ Adam momentum + variance (FP32) │
│ Master Weights     │ 4 bytes             │ FP32 copy for optimizer update  │
├─────────────────────────────────────────────────────────────────────────────┤
│ TOTAL              │ 16 bytes/param      │ ~16 GB per billion parameters   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Memory formula**:
```
Model/Optimizer Memory ≈ num_params × 16 bytes
```

### Activation Memory

Activations scale with batch size, sequence length, and model dimensions:

```
Per-layer activation memory (approximate, with FlashAttention):

Activations ≈ batch × seq_len × hidden_dim × bytes_per_element × factor

Where factor accounts for:
- Input/output of each sublayer (attention, FFN)
- Intermediate FFN activations (typically 4× hidden_dim)
- Normalization layers
- Residual connections

Typical factor: 10-20× for full activations, 2-4× with selective checkpointing
```

**Activation estimates for 8K context, batch=1, BF16**:

| Model Size | Hidden Dim | Layers | Activations (full) | Activations (selective AC) |
|------------|------------|--------|-------------------|---------------------------|
| 8B | 4096 | 32 | ~40 GB | ~10 GB |
| 70B | 8192 | 80 | ~200 GB | ~50 GB |
| 405B | 16384 | 126 | ~600 GB | ~150 GB |

### Llama3 Model Specifications

Reference architecture parameters:

| Model | Parameters | Hidden Dim | Layers | Heads | KV Heads | FFN Dim |
|-------|------------|------------|--------|-------|----------|---------|
| 8B | 8B | 4096 | 32 | 32 | 8 | ~14K |
| 70B | 70B | 8192 | 80 | 64 | 8 | ~28K |
| 405B | 405B | 16384 | 126 | 128 | 8 | ~53K |

### Calculation: When Each Parallelism is Needed

#### Single GPU (No Parallelism)

**Capacity**: 192 GB B200

**Maximum model size**:
```
192 GB available
─ ~20 GB reserved (CUDA, framework overhead)
= ~170 GB usable

170 GB ÷ 16 bytes/param = ~10.6B parameters (model + optimizer)
Minus activations (~10 GB with AC) = ~10B parameters max
```

**Conclusion**: Models up to ~8-10B can fit on a single B200 with activation
checkpointing. In practice, use FSDP even for 8B for memory headroom.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Single GPU: Models ≤ 8B (tight), practical limit ~5-6B for comfortable fit │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### FSDP on Single Node (8 GPUs)

**Capacity**: 8 × 192 GB = 1.5 TB (shared via FSDP)

With FSDP, each GPU holds 1/8 of parameters and optimizer states, but needs
memory for:
- Full gradients during backward (temporarily)
- Local activations

**Effective capacity per GPU**:
```
Parameters + Optimizer: (P × 16) ÷ 8 = 2 bytes/param per GPU
Gradients (temporary): P × 2 bytes (full, during backward)
Activations: ~10-50 GB depending on model size and AC

Example for 70B model:
- Params + Optimizer: 70B × 2 = 140 GB distributed across 8 GPUs = 17.5 GB/GPU
- Gradients: 70B × 2 = 140 GB (temporary, during backward)
- Activations: ~50 GB with selective AC

Total per GPU: 17.5 + 140/8 + 50 ≈ 85 GB — fits in 192 GB!
Wait, gradients are also sharded in FSDP2 reduce-scatter...

Corrected with FSDP2:
- Sharded params + opt: 17.5 GB
- Sharded gradients: 17.5 GB
- Activations: ~50 GB
- Temporary all-gathered params: ~17.5 GB (during forward/backward)
Total: ~100 GB — fits comfortably!
```

**Maximum model size with 8-way FSDP**:
```
Usable per GPU: ~170 GB
With activations (~50 GB): ~120 GB for model/optimizer shards

Each GPU needs: (P × 16) ÷ 8 + temporary overhead ≈ (P × 4) per GPU
120 GB = P × 4 bytes → P ≈ 30B effective
But with activation scaling, practical limit is ~70B
```

**Conclusion**: 8-way FSDP on single node handles models up to ~70B.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FSDP (8 GPUs): Models 8B-70B                                                │
│ - 8B: Comfortable, can increase batch size                                  │
│ - 40B: Good fit, moderate batch size                                        │
│ - 70B: Tight fit, batch=1-2, requires selective AC                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### HSDP (Multi-Node FSDP + DDP)

When to use HSDP instead of pure FSDP across nodes:

**Benefits of HSDP over cross-node FSDP**:
1. FSDP communication (all-gather, reduce-scatter) stays on fast NVLink
2. Only gradient averaging (DDP) goes over slower inter-node network
3. Better throughput for models that fit in single-node FSDP

**Configuration**:
```toml
[parallelism]
data_parallel_replicate_degree = N  # Number of nodes (DDP across nodes)
data_parallel_shard_degree = 8      # FSDP within each node
```

**When HSDP helps**:
- Model fits in 8-way FSDP (≤70B)
- Training on 2+ nodes
- Want maximum throughput with data parallelism

**When to use pure FSDP across nodes instead**:
- Model too large for single-node FSDP (>70B)
- Memory per GPU is the bottleneck

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HSDP: Models 8B-70B on 2+ nodes                                             │
│ - Combines throughput scaling (DDP) with memory efficiency (FSDP)          │
│ - Best for models that fit in single-node FSDP                             │
│                                                                             │
│ Example configs:                                                            │
│ - 8B on 4 nodes:  dp_replicate=4, dp_shard=8 (32 GPUs)                     │
│ - 70B on 4 nodes: dp_replicate=4, dp_shard=8 (32 GPUs, batch=1-2)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### When Tensor Parallelism (TP) is Required

TP becomes necessary when:

1. **Single layer exceeds GPU memory** (even with FSDP)
2. **Activation memory per GPU is too high**
3. **Improving compute efficiency for very large layers**

**Layer size analysis**:

For a transformer layer, the largest components are:
```
Attention weights: 4 × dim × dim (wq, wk, wv, wo simplified)
FFN weights: 3 × dim × ffn_dim

Llama3 70B (dim=8192, ffn_dim≈28K):
- Attention: 4 × 8192 × 8192 = 268M params = 536 MB (BF16)
- FFN: 3 × 8192 × 28K = 688M params = 1.4 GB (BF16)
- Per layer: ~2 GB in BF16

Llama3 405B (dim=16384, ffn_dim≈53K):
- Attention: 4 × 16384 × 16384 = 1.07B params = 2.1 GB (BF16)
- FFN: 3 × 16384 × 53K = 2.6B params = 5.2 GB (BF16)
- Per layer: ~7.3 GB in BF16
```

With FSDP, layers are sharded, so individual layer size rarely causes OOM.
But **activation memory** for large models benefits significantly from TP:

```
Activation memory reduction with TP:

Without TP: Each GPU stores full [batch, seq, hidden] activations
With TP=8: Each GPU stores [batch, seq, hidden/8] for some tensors

For 405B (hidden=16384), seq=8K, batch=1:
- Full activation tensor: 1 × 8K × 16K × 2 = 256 MB per tensor
- With TP=8: 1 × 8K × 2K × 2 = 32 MB per tensor

Per-layer savings add up across 126 layers!
```

**When to enable TP**:

| Model Size | Hidden Dim | TP Recommendation | Reason |
|------------|------------|-------------------|--------|
| ≤8B | ≤4096 | TP=1 (disabled) | FSDP sufficient |
| 8B-40B | 4096-6144 | TP=1 or TP=2 | Optional for activation savings |
| 70B | 8192 | TP=2 or TP=4 | Reduces activation memory |
| 200B+ | 12288+ | TP=4 or TP=8 | Required for activation memory |
| 405B | 16384 | TP=8 | Required for practical training |

**TP constraints reminder**: `n_kv_heads % tp == 0` (Llama3 has 8 KV heads,
so TP must be 1, 2, 4, or 8)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tensor Parallelism Guidelines:                                              │
│                                                                             │
│ - ≤70B: TP optional, use TP=2-4 if activation memory is tight              │
│ - 70B-200B: TP=4 recommended                                                │
│ - 200B+: TP=8 required                                                      │
│                                                                             │
│ Rule of thumb: Enable TP when hidden_dim > 8192 or params > 70B            │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### When Pipeline Parallelism (PP) is Required

PP is the last resort when FSDP + TP isn't sufficient:

**PP memory calculation**:
```
With PP, each GPU holds only (n_layers ÷ pp_degree) layers.

Example: 405B with PP=4
- 126 layers ÷ 4 = ~31 layers per stage
- Each stage: 405B ÷ 4 ≈ 100B parameters
- Memory per stage: 100B × 16 bytes = 1.6 TB

Still need FSDP to shard each stage!

With PP=4, TP=8, FSDP=8 on 256 GPUs (32 nodes):
- PP splits into 4 stages
- Each stage: TP=8 for layers, FSDP=8 for sharding
- Per GPU: 405B ÷ (4 × 8 × 8) = ~1.6B params effective
- Memory: ~25 GB for model/optimizer — very comfortable!
```

**When PP is needed**:
1. Model doesn't fit even with FSDP + TP across available GPUs
2. Want to scale beyond TP limits (TP=8 max for Llama3)
3. Training 200B+ models efficiently

**PP considerations**:
- Adds pipeline bubbles (reduced efficiency)
- Requires careful microbatch sizing
- More complex checkpoint/resume

| Model Size | Minimum Configuration | Recommended Configuration |
|------------|----------------------|--------------------------|
| 405B | PP=2, TP=8, FSDP≥8 | PP=4, TP=8, FSDP=8 |
| 1T+ | PP=4+, TP=8, FSDP≥8 | PP=8+, TP=8, FSDP=8 |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Pipeline Parallelism Guidelines:                                            │
│                                                                             │
│ - ≤200B: Usually not needed (FSDP + TP sufficient)                         │
│ - 200B-500B: PP=2-4 may help with memory or scaling                        │
│ - 500B+: PP required (PP=4-8)                                              │
│                                                                             │
│ Rule of thumb: Use PP when FSDP + TP can't fit the model                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complete Decision Flowchart

```
                    ┌─────────────────────────┐
                    │   Model Size (params)   │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
       ┌─────────┐         ┌─────────┐         ┌─────────┐
       │  ≤10B   │         │ 10B-70B │         │  >70B   │
       └────┬────┘         └────┬────┘         └────┬────┘
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │ Single GPU or │   │    FSDP or    │   │  FSDP + TP    │
    │  FSDP (8 GPU) │   │     HSDP      │   │   required    │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │ Multi-node?   │   │ Multi-node?   │   │    >200B?     │
    │   Use HSDP    │   │   Use HSDP    │   │   Add PP      │
    └───────────────┘   └───────────────┘   └───────────────┘
```

### Recommended Configurations by Model Size

**B200 Cluster (8 GPUs/node, 192 GB/GPU, 8K context)**:

| Model | Nodes | GPUs | dp_replicate | dp_shard | TP | PP | Notes |
|-------|-------|------|--------------|----------|----|----|-------|
| 8B | 1 | 8 | 1 | 8 | 1 | 1 | Pure FSDP, batch=8-16 |
| 8B | 4 | 32 | 4 | 8 | 1 | 1 | HSDP for throughput |
| 40B | 1 | 8 | 1 | 8 | 1 | 1 | FSDP, batch=2-4 |
| 40B | 4 | 32 | 4 | 8 | 1 | 1 | HSDP, batch=2-4 |
| 70B | 1 | 8 | 1 | 8 | 2 | 1 | FSDP+TP, batch=1-2 |
| 70B | 4 | 32 | 4 | 4 | 2 | 1 | HSDP+TP, batch=1-2 |
| 70B | 8 | 64 | 2 | 4 | 8 | 1 | High TP for memory |
| 200B | 8 | 64 | 1 | 8 | 8 | 1 | FSDP+TP, batch=1 |
| 405B | 16 | 128 | 1 | 4 | 8 | 4 | Full 3D parallelism |
| 405B | 32 | 256 | 2 | 4 | 8 | 4 | HSDP+TP+PP |

**Configuration formulas**:
```
world_size = dp_replicate × dp_shard × tp × pp
nodes = world_size ÷ 8

Memory per GPU ≈ (params × 16) ÷ (dp_shard × tp × pp) + activations
```

### Example Calculation: 70B on 4 Nodes

**Setup**: 70B Llama3, 4 nodes × 8 B200 GPUs = 32 GPUs

**Option 1: Pure HSDP (dp_replicate=4, dp_shard=8)**
```
Model memory: 70B × 16 bytes = 1.12 TB
Sharded across 8 GPUs: 1.12 TB ÷ 8 = 140 GB/GPU
Activations (8K context, selective AC): ~50 GB
Total: ~190 GB — just barely fits 192 GB!

Problem: No headroom for batch size > 1
```

**Option 2: HSDP + TP (dp_replicate=4, dp_shard=4, tp=2)**
```
Effective sharding: dp_shard × tp = 4 × 2 = 8-way
Model memory per GPU: 1.12 TB ÷ 8 = 140 GB
Activations with TP=2: ~35 GB (reduced by TP)
Total: ~175 GB — fits with headroom!

Benefits: Can use batch=2, better activation memory
```

**Option 3: More TP (dp_replicate=2, dp_shard=2, tp=8)**
```
Effective sharding: dp_shard × tp = 2 × 8 = 16-way
Model memory per GPU: 1.12 TB ÷ 16 = 70 GB
Activations with TP=8: ~15 GB
Total: ~85 GB — very comfortable!

Trade-off: Less data parallelism (only 4 replicas)
```

**Recommendation for 70B on 32 GPUs**: Option 2 (HSDP + TP=2) balances
memory efficiency with training throughput.

### Summary: Parallelism Thresholds for B200 (192 GB)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Parallelism Selection Summary                           │
│                   (B200 GPUs, 8K context, BF16 training)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model Size        Parallelism Stack           Key Constraint              │
│  ───────────       ──────────────────          ──────────────              │
│                                                                             │
│  ≤8B               FSDP (single node)          Memory comfortable          │
│                    or HSDP (multi-node)                                     │
│                                                                             │
│  8B-40B            FSDP or HSDP                 Good batch sizes           │
│                                                                             │
│  40B-70B           FSDP/HSDP + TP=2            Activation memory           │
│                                                                             │
│  70B-200B          FSDP + TP=4-8               TP required for activations │
│                                                                             │
│  200B-500B         FSDP + TP=8 (+ PP=2-4)      May need PP for model fit   │
│                                                                             │
│  >500B             FSDP + TP=8 + PP=4-8        Full 3D parallelism         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Quick Reference:                                                           │
│  • FSDP alone: ≤70B (single node), ≤40B comfortable                        │
│  • Add TP: When hidden_dim > 8K or activations tight                       │
│  • Add PP: When model > 200B or FSDP+TP can't fit                          │
│  • HSDP over FSDP: When model fits in single-node FSDP                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Single GPU and Debug Configurations

### Running Without Any Parallelism

All parallelism can be disabled for single GPU runs. TorchTitan provides
debug model configs for this purpose.

**Example debug config** (`debug_model.toml`):

```toml
[job]
description = "Debug model for single GPU testing"

[model]
name = "llama3"
flavor = "debugmodel"

[training]
local_batch_size = 8
seq_len = 2048
steps = 10

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1   # Auto = 1 on single GPU
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
context_parallel_degree = 1
```

**Running**:
```bash
NGPU=1 ./run_train.sh --job.config_file=path/to/debug_model.toml
```

### Communication Modes for Debugging

TorchTitan supports special communication modes for debugging:

```toml
[comm]
mode = "default"      # Normal distributed training
# mode = "local_tensor"  # Debug: simulates distributed on single process
# mode = "fake_backend"  # Dry run: validates config without GPU
```

**Usage**:
```bash
# Config validation without GPU
COMM_MODE="fake_backend" ./run_train.sh

# Debug mode (single process, sequential ranks)
COMM_MODE="local_tensor" ./run_train.sh
```

### Where to Find Debug Configs

Each model has a `debug_model.toml`:
- `torchtitan/models/llama3/train_configs/debug_model.toml`
- `torchtitan/models/llama4/train_configs/debug_model.toml`
- `torchtitan/models/deepseek_v3/train_configs/debug_model.toml`
- etc.

---

## Seed Checkpoints for Reproducibility

### What is a Seed Checkpoint?

A seed checkpoint is a full, unsharded model checkpoint created on a single
device. It allows reproducible training across different parallelism
configurations.

### Creating a Seed Checkpoint

```bash
NGPU=1 ./run_train.sh \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint \
  --parallelism.data_parallel_replicate_degree=1 \
  --parallelism.data_parallel_shard_degree=1 \
  --parallelism.tensor_parallel_degree=1 \
  --parallelism.pipeline_parallel_degree=1
```

**Requirements**:
- `WORLD_SIZE=1` (single device only)
- `checkpoint.enable=true`
- All parallelism degrees set to 1

**What happens**:
1. Model is initialized on CPU without parallelism
2. Full unsharded weights are saved at step 0
3. DCP format allows automatic resharding when loaded

### Loading Seed Checkpoint on Multi-GPU

```bash
NGPU=8 ./run_train.sh \
  --checkpoint.initial_load_path=/path/to/seed/checkpoint \
  --parallelism.data_parallel_shard_degree=4 \
  --parallelism.tensor_parallel_degree=2
```

DCP automatically reshards the seed checkpoint to match the new parallelism
configuration.

---

## Fault Tolerance (TorchFT)

### Overview

TorchTitan integrates with TorchFT for elastic fault-tolerant training. When
enabled, training can survive node failures and continue with remaining nodes.

### How FT Works with HSDP

FT uses HSDP (Hybrid Sharded Data Parallel) architecture:

```
Fault Tolerance with HSDP:

┌─────────────────────────────────────────────────────────────────────┐
│                    TorchFT Replica Management                       │
│                                                                     │
│   Replica 0 (group_size=4)         Replica 1 (group_size=4)        │
│   ┌─────────────────────┐          ┌─────────────────────┐         │
│   │ G0  G1  G2  G3      │          │ G4  G5  G6  G7      │         │
│   │ (FSDP within)       │          │ (FSDP within)       │         │
│   └─────────────────────┘          └─────────────────────┘         │
│            │                                │                       │
│            └────── Gradient sync ───────────┘                       │
│                  (managed by TorchFT)                               │
│                                                                     │
│   If Replica 1 fails:                                               │
│   - Replica 0 continues training                                    │
│   - Gradient sync skipped for failed replica                        │
│   - When Replica 1 rejoins, state is synchronized                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration

```toml
[fault_tolerance]
enable = true
process_group = "gloo"          # or "nccl"
replica_id = 0                  # This run's replica ID
group_size = 4                  # GPUs per replica
min_replica_size = 1            # Minimum replicas to continue

# Optional semi-synchronous training
semi_sync_method = "diloco"     # or "local_sgd"
sync_steps = 5                  # Steps between synchronization
```

### Checkpoint Strategy with FT

FT uses dual checkpointing:

1. **Full checkpoint**: Saved by one rank per replica group (model, optimizer,
   scheduler, train state)
2. **Per-replica checkpoint**: Saved by all replicas (dataloader state only)

```
Checkpoint structure with FT:

{dump_folder}/checkpoint/
├── step-1000/                    # Full checkpoint (shared)
│   └── *.distcp
├── ft-replica-0/                 # Replica 0 dataloader state
│   └── step-1000/
└── ft-replica-1/                 # Replica 1 dataloader state
    └── step-1000/
```
