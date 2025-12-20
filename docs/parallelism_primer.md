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
