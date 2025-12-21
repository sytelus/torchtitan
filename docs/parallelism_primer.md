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

### How CP Works

```
Context Parallel Sequence Sharding (cp=4, seq_len=8192):

Original sequence: [token_0, token_1, ..., token_8191]
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GPU 0: tokens [0:2048]        ─┐                                     │
│ GPU 1: tokens [2048:4096]      │── Ring attention pattern           │
│ GPU 2: tokens [4096:6144]      │   KV are rotated between ranks     │
│ GPU 3: tokens [6144:8192]     ─┘                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### Ring Attention Pattern

During attention computation, each GPU needs to attend to all tokens, but only
holds a subset. CP uses a "ring" communication pattern:

```
Ring Attention Communication (cp=4):

Step 1: Each GPU computes local Q×K^T for its own KV
Step 2: Rotate KV shards around the ring
Step 3: Each GPU computes Q×K^T for received KV
Step 4: Repeat until all KV have been seen

┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ GPU0 │───►│ GPU1 │───►│ GPU2 │───►│ GPU3 │
│ KV_0 │    │ KV_1 │    │ KV_2 │    │ KV_3 │
└──────┘    └──────┘    └──────┘    └──────┘
    ▲                                   │
    └───────────────────────────────────┘
              Ring rotation
```

### CP Configuration

```toml
[parallelism]
context_parallel_degree = 4
context_parallel_rotate_method = "allgather"  # or "alltoall"
```

**Rotation methods**:
- `"allgather"`: All-gather all KV shards after first sub-SDPA (default)
- `"alltoall"`: All-to-all shuffle KV shards for more balanced communication

### CP + TP + FSDP Mesh Topology Example

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

### Sequence Length Constraint

CP requires sequence length to be divisible by `2 × cp_degree` (for load
balancing):

```python
seq_len_divisor = tp * (cp * 2)
assert seq_len % seq_len_divisor == 0
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
