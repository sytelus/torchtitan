# Configuration Reference

This document describes every option in `JobConfig` (the main config object used
by `torchtitan/train.py`). The format below uses TOML section keys and includes
defaults. CLI overrides use `--section.key`.

Example:

```
[training]
local_batch_size = 8
```

CLI equivalent:

```
--training.local_batch_size=8
```

## Precedence

1. CLI arguments
2. TOML config
3. `JobConfig` defaults

## Global notes

- Comma-separated lists (e.g., `model.converters`) can be passed as
  `--model.converters="a,b,c"`.
- If a field is missing from TOML, its default is used.
- Invalid keys in TOML will raise an error.

---

## [job]

- `config_file` (default: null)
  Path to the TOML config file. Usually passed via `--job.config_file`.

- `dump_folder` (default: "./outputs")
  Base folder for outputs: checkpoints, logs, traces, and profiling artifacts.

- `description` (default: "default job")
  A human-readable label for logging.

- `print_config` (default: false)
  If true, print the resolved config to the terminal.

- `save_config_file` (default: null)
  If set, writes the resolved config JSON into `dump_folder/`.

- `custom_config_module` (default: "")
  Dotted import path to a module that defines a custom `JobConfig` extension.
  Useful for adding new config sections.

---

## [profiling]

- `enable_profiling` (default: false)
  Enable PyTorch profiler during training.

- `save_traces_folder` (default: "profile_traces")
  Subfolder under `dump_folder` for profiler traces.

- `profile_freq` (default: 10)
  Profile every N training steps.

- `profiler_active` (default: 1)
  Active steps per profiling cycle. Used by `torch.profiler.schedule()`.

- `profiler_warmup` (default: 3)
  Warmup steps per profiling cycle.

- `enable_memory_snapshot` (default: false)
  Enable memory snapshot capture during training.

- `save_memory_snapshot_folder` (default: "memory_snapshot")
  Subfolder under `dump_folder` for memory snapshots.

---

## [metrics]

- `log_freq` (default: 10)
  Log metrics every N training steps.

- `enable_tensorboard` (default: false)
  Enable TensorBoard logging.

- `disable_color_printing` (default: false)
  Disable ANSI color codes in logs.

- `save_tb_folder` (default: "tb")
  Subfolder under `dump_folder` for TensorBoard files.

- `save_for_all_ranks` (default: false)
  If false, only the rank that owns the loss logs metrics. Set true to log
  on all ranks (useful for debugging).

- `enable_wandb` (default: false)
  Enable Weights & Biases logging.

---

## [model]

- `name` (default: "llama3")
  Model family name. Used to select the `TrainSpec`.

- `flavor` (default: "debugmodel")
  Model size/config variant (e.g., "8B", "70B").

- `hf_assets_path` (default: "./tests/assets/tokenizer")
  Path to Hugging Face assets directory: weights (safetensors), config.json,
  tokenizer, etc. Used by tokenizer/model conversion and checkpoint adapters.

- `tokenizer_path` (default: null)
  Deprecated. Use `hf_assets_path`.

- `converters` (default: [])
  List of model converters to apply before parallelism. Example:
  `quantize.linear.float8` swaps `nn.Linear` to `Float8Linear`.

- `print_after_conversion` (default: false)
  Print the model after applying converters.

---

## [optimizer]

- `name` (default: "AdamW")
  Optimizer class name.

- `lr` (default: 8e-4)
  Base learning rate.

- `beta1` (default: 0.9)
- `beta2` (default: 0.95)
  Adam beta coefficients.

- `eps` (default: 1e-8)
  Adam epsilon.

- `weight_decay` (default: 0.1)
  Weight decay coefficient.

- `implementation` (default: "fused")
  Optimizer implementation: `fused`, `foreach`, or `for-loop`.

- `early_step_in_backward` (default: false)
  Apply optimizer step inside backward pass. Not compatible with gradient
  clipping or post-accumulate hooks.

---

## [lr_scheduler]

- `warmup_steps` (default: 200)
  Warmup steps before decay.

- `decay_ratio` (default: null)
  If set, use Warmup-Stable-Decay: decay only in the last `decay_ratio` of
  total steps. If null, decay starts immediately after warmup.

- `decay_type` (default: "linear")
  Decay type: `linear`, `sqrt`, or `cosine`.

- `min_lr_factor` (default: 0.0)
  Lower bound for LR as a fraction of initial LR.

---

## [training]

- `dataset` (default: "c4_test")
  Dataset name (e.g., `c4`).

- `dataset_path` (default: null)
  Local filesystem path to dataset. Overrides download logic.

- `local_batch_size` (default: 8)
  Batch size per device (before gradient accumulation).

- `global_batch_size` (default: -1)
  Total batch size across DP ranks and gradient accumulation. `-1` means
  `local_batch_size * dp_degree` (one accumulation step).

- `seq_len` (default: 2048)
  Sequence length for training batches.

- `max_norm` (default: 1.0)
  Gradient clipping norm. Set to 0 to disable clipping.

- `steps` (default: 10000)
  Total training steps.

- `enable_cpu_offload` (default: false)
  Enable FSDP CPU offloading for params/grad/optimizer state.

- `dtype` (default: "float32")
  *Full* model dtype. Use `bfloat16` for full BF16 training.

- `mixed_precision_param` (default: "bfloat16")
  Parameter dtype when mixed precision is enabled (via FSDP or AMP).

- `mixed_precision_reduce` (default: "float32")
  Reduction dtype in FSDP.

- `gc_freq` (default: 50)
  Force Python GC every N steps.

- `gc_debug` (default: false)
  Aggressively call `gc.collect()` every step to catch reference cycles.

### [training.dataloader]

- `num_workers` (default: 0)
- `persistent_workers` (default: false)
- `pin_memory` (default: false)
- `prefetch_factor` (default: null)

These map directly to `torch.utils.data.DataLoader` (used via
StatefulDataLoader). `persistent_workers` and `prefetch_factor` require
`num_workers > 0`.

#### Why `pin_memory` is False by Default

When `pin_memory=true`, PyTorch allocates data in "pinned" (page-locked) CPU
memory, allowing faster CPU→GPU transfers via DMA (Direct Memory Access)
without involving the CPU.

The default is `false` because:

1. **Memory Overhead**: Pinned memory is non-swappable and stays locked in RAM.
   With large batch sizes or many workers, this can consume significant CPU
   memory.

2. **FSDP CPU Offloading Conflict**: When `enable_cpu_offload=true`, FSDP moves
   parameters and optimizer states to CPU. If both the dataloader and FSDP
   compete for pinned memory, you can run out of CPU RAM quickly.

3. **Not Always Beneficial**: The benefit of pinned memory depends on how
   CPU-bound your data loading is and whether data loading overlaps with GPU
   compute. With `num_workers=0` (default), data loading is synchronous, and
   pinned memory has limited benefit.

**When to enable it**: Enable when you have sufficient CPU RAM and want to
optimize CPU→GPU transfer for multi-worker dataloading:

```toml
[training.dataloader]
num_workers = 4
pin_memory = true
persistent_workers = true
prefetch_factor = 2
```

---

## [parallelism]

- `data_parallel_replicate_degree` (default: 1)
  DDP/HSDP replication degree. >1 enables DDP or HSDP.

- `data_parallel_shard_degree` (default: -1)
  FSDP/HSDP sharding degree. `-1` uses leftover ranks.

- `fsdp_reshard_after_forward` (default: "default")
  Policy for resharding after forward: `default`, `always`, `never`.

- `tensor_parallel_degree` (default: 1)
  TP degree.

- `disable_loss_parallel` (default: false)
  Disable loss-parallel when TP is enabled.

- `enable_async_tensor_parallel` (default: false)
  Enable async TP collectives (effective with `torch.compile`).

- `pipeline_parallel_degree` (default: 1)
  PP degree (number of ranks). For looped schedules, still the number of
  physical ranks.

- `module_fqns_per_model_part` (default: null)
  Explicit module lists for PP stage partitioning.

- `pipeline_parallel_first_stage_less_layers` (default: 1)
  Reduce layer count for first stage to account for embeddings.

- `pipeline_parallel_last_stage_less_layers` (default: 1)
  Reduce layer count for last stage to account for output layers.

- `pipeline_parallel_layers_per_stage` (default: null)
  If set, split layers into this many per stage (virtual stages).

- `pipeline_parallel_schedule` (default: "1F1B")
  Pipeline schedule name. Must be compatible with stage layout.

- `pipeline_parallel_schedule_csv` (default: "")
  Path to a CSV schedule. Only valid with schedule types that support it.

- `pipeline_parallel_microbatch_size` (default: 1)
  Microbatch size for PP. Must divide `local_batch_size`.

- `pipeline_parallel_expert_parallel_overlap` (default: true)
  Enable overlap between EP and PP (DualPipeV only).

- `context_parallel_degree` (default: 1)
  Context parallel degree.

- `context_parallel_rotate_method` (default: "allgather")
  Communication method for CP KV exchange: `allgather` or `alltoall`.

- `expert_parallel_degree` (default: 1)
  EP degree for MoE models.

- `expert_tensor_parallel_degree` (default: 1)
  ETP degree for MoE experts.

- `expert_parallel_comm_backend` (default: "standard")
  Expert-parallel communication backend: `standard` or `deepep`.

---

## [checkpoint]

- `enable` (default: false)
  Enable checkpointing.

- `enable_ft_dataloader_checkpoints` (default: true)
  Save dataloader state for fault-tolerant (TorchFT) runs.

- `folder` (default: "checkpoint")
  Subfolder under `dump_folder` for checkpoints.

- `interval` (default: 500)
  Checkpoint every N steps.

- `initial_load_path` (default: null)
  Load initial checkpoint from a different run/path.

- `initial_load_model_only` (default: true)
  If true, load only model weights from `initial_load_path`.

- `initial_load_in_hf` (default: false)
  Load HF safetensors format (model-only).

- `initial_load_in_hf_quantized` (default: false)
  Load HF safetensors with quantized keys (requires a suitable adapter).

- `last_save_model_only` (default: true)
  Save only model weights at final step.

- `last_save_in_hf` (default: false)
  Save final model in HF safetensors format (requires consolidation).

- `export_dtype` (default: "float32")
  Dtype for the final exported checkpoint when `last_save_model_only=true`.

- `async_mode` (default: "disabled")
  Async checkpointing mode: `disabled`, `async`, `async_with_pinned_mem`.

- `keep_latest_k` (default: 10)
  Keep only the latest K checkpoints (0 keeps all).

- `load_step` (default: -1)
  Step to load. `-1` loads the latest checkpoint.

- `exclude_from_loading` (default: [])
  List of checkpoint keys to skip (e.g., `optimizer,lr_scheduler`).

- `enable_first_step_checkpoint` (default: false)
  Save an early checkpoint after step 1 (sanity check).

- `create_seed_checkpoint` (default: false)
  Initialize full model and save a seed checkpoint (must be single device).

- `load_only` (default: false)
  Load checkpoints but do not save during the run.

---

## [activation_checkpoint]

- `mode` (default: "selective")
  `none`, `selective`, `full`, or `memory_budget`.

- `selective_ac_option` (default: "2")
  For selective mode: `op` (op-level policy) or an integer string for
  layer-frequency (e.g., `"2"` means every 2nd layer).

- `per_op_sac_force_recompute_mm_shapes_by_fqns` (default: ["moe.router.gate"])
  Force recompute for matmul shapes matching these module FQNs.

- `early_stop` (default: false)
  Stop recomputing if all activations are already rematerialized.

- `memory_budget` (default: 0.5)
  Used only in `memory_budget` mode (requires compile).

- `visualize_memory_budget_pareto` (default: false)
  Generates an SVG visualization of runtime vs. activation memory tradeoffs.
  The visualization evaluates all memory budget values from 0.0 to 1.0 in
  increments of 0.05, helping you choose the optimal `memory_budget` value.
  Output is saved to `{job.dump_folder}/memory_budget_pareto/`. See
  [modelling.md](modelling.md#visualizing-memory-budget-pareto-frontier) for usage details.

- `preserve_rng_state` (default: false)
  Preserve RNG state for deterministic recomputation.

- `determinism_check` (default: "default")
  Determinism setting for checkpointing (see PyTorch docs).

- `debug` (default: false)
  Enable debug info for checkpointing.

### Understanding Activation Checkpointing Modes

Activation checkpointing (AC), also known as gradient checkpointing, trades
compute for memory. Instead of storing all intermediate activations during
forward (for use in backward), AC discards them and recomputes during backward.

**Memory vs Compute Tradeoff:**
- Without AC: O(n_layers) activation memory, no extra compute
- With AC: O(1) to O(sqrt(n_layers)) memory, ~10-33% more compute

#### The Four Modes

| Mode | Description | Memory Savings | Compute Overhead |
|------|-------------|----------------|------------------|
| `none` | No checkpointing | None | None |
| `selective` | Checkpoint based on policy | Medium-High | Low-Medium |
| `full` | Checkpoint every block | Maximum | ~33% |
| `memory_budget` | Compiler-guided (requires torch.compile) | Configurable | Optimized |

#### Selective Mode: Layer-Frequency vs Op-Level

The `selective_ac_option` controls how selective checkpointing works:

**Layer-Frequency (`"2"`, `"3"`, etc.):**
```
selective_ac_option = "2"  # Checkpoint every 2nd transformer block

Layer 0: Save activations
Layer 1: CHECKPOINT (discard, recompute in backward)
Layer 2: Save activations
Layer 3: CHECKPOINT (discard, recompute in backward)
...
```

**Op-Level (`"op"`):**
```
selective_ac_option = "op"  # Fine-grained per-operation policy

For each layer:
  - matmul (mm): SAVE (expensive to recompute)
  - attention (SDPA): SAVE (very expensive)
  - reduce_scatter: SAVE (communication op)
  - max: SAVE (for FP8 scaling)
  - layer_norm: RECOMPUTE (cheap)
  - activations: RECOMPUTE (cheap)
  - dropout: RECOMPUTE (cheap)
```

#### Why Defaults Use `"2"` but Production Uses `"op"`

| Setting | Default | Llama 8B Production |
|---------|---------|---------------------|
| `mode` | `"selective"` | `"selective"` |
| `selective_ac_option` | `"2"` | `"op"` |

**Default (`"2"`) rationale:**
- Model-agnostic: works without model-specific op lists
- Simpler and more predictable behavior
- Good for debugging and quick experiments

**Production (`"op"`) rationale:**
- Better memory/compute tradeoff: only recomputes cheap ops
- FP8 compatible: saves `max` ops for scaling factor computation
- Distributed-safe: saves communication ops like `reduce_scatter`

The op-level policy requires a model-specific `_op_sac_save_list` that defines
which operations are expensive and should be saved. Each model in TorchTitan
defines its own list in its `parallelize.py`.

#### Recommendations

**For production Llama/Qwen training:**
```toml
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
```

**For debugging or quick experiments:**
```toml
[activation_checkpoint]
mode = "selective"
selective_ac_option = "2"
```

**For maximum memory savings (at compute cost):**
```toml
[activation_checkpoint]
mode = "full"
```

**For compiler-optimized checkpointing (requires torch.compile):**
```toml
[activation_checkpoint]
mode = "memory_budget"
memory_budget = 0.5  # 0.0 = full AC, 1.0 = no AC

[compile]
enable = true
```

---

## [compile]

- `enable` (default: false)
  Enable `torch.compile` for selected components.

- `components` (default: ["model", "loss"])
  Which components to compile.

- `backend` (default: "inductor")
  Torch compile backend.

---

## [quantize]

### [quantize.linear.float8]

- `enable_fsdp_float8_all_gather` (default: false)
  Communicate FSDP all-gathers in FP8 (tensorwise scaling only).

- `precompute_float8_dynamic_scale_for_fsdp` (default: false)
  Precompute FP8 scales for all parameters to reduce per-param communication.

- `recipe_name` (default: null)
  Use a TorchAO recipe (e.g., `tensorwise`, `rowwise`).

- `filter_fqns` (default: [])
  List of module FQNs to skip FP8 conversion.

- `emulate` (default: false)
  Emulate FP8 on unsupported hardware (eager mode only).

### [quantize.grouped_mm.float8]

- `fqns` (default: [])
  MoE module FQNs to apply grouped FP8 GEMMs.

### [quantize.linear.mx]

- `mxfp8_dim1_cast_kernel_choice` (default: "triton")
  Kernel for dim1 cast (performance tuning).

- `recipe_name` (default: "mxfp8_cublas")
  MX recipe name (see TorchAO).

- `filter_fqns` (default: ["output"])
  Module FQNs to skip MX conversion.

### [quantize.grouped_mm.mx]

- `recipe_name` (default: "mxfp8")
  MX recipe for grouped GEMMs.

- `fqns` (default: [])
  MoE module FQNs to apply MX grouped GEMMs.

---

## [comm]

- `init_timeout_seconds` (default: 300)
  Timeout for process group initialization and first step.

- `train_timeout_seconds` (default: 100)
  Timeout for collectives after step 1.

- `trace_buf_size` (default: 20000)
  Flight recorder ring buffer size (0 disables).

- `save_traces_folder` (default: "comm_traces")
  Subfolder under `dump_folder` for comm traces.

- `save_traces_file_prefix` (default: "rank_")
  Prefix for trace files.

- `mode` (default: "default")
  Communication mode: `default`, `fake_backend`, `local_tensor`.

---

## [memory_estimation]

- `enable` (default: false)
  Enable memory estimation (FSDP).

- `disable_fake_mode` (default: false)
  Disable FakeTensorMode in estimation.

---

## [fault_tolerance]

- `enable` (default: false)
  Enable TorchFT integration (uses HSDP).

- `process_group` (default: "gloo")
  Process group backend for TorchFT.

- `process_group_timeout_ms` (default: 10000)
  Timeout for TorchFT process group.

- `replica_id` (default: 0)
  TorchFT replica id.

- `group_size` (default: 0)
  Number of TorchFT replicate groups.

- `min_replica_size` (default: 1)
  Minimum replicas per step.

- `semi_sync_method` (default: null)
  Semi-sync method: `local_sgd`, `diloco`, etc.

---

## [experimental]

- `custom_import` (default: "")
  Dotted import path to load custom modules at startup.

- `custom_args_module` (default: "")
  Deprecated. Use `job.custom_config_module` instead.

---

## [validation]

- `enable` (default: false)
  Enable validation.

- `dataset` (default: "c4_validation")
  Validation dataset name.

- `dataset_path` (default: null)
  Path to validation dataset.

- `local_batch_size` (default: 8)
  Validation batch size.

- `seq_len` (default: 2048)
  Validation sequence length.

- `freq` (default: 10)
  Validate every N steps.

- `steps` (default: -1)
  Number of validation steps (`-1` means exhaust dataset; beware of hangs
  if ranks diverge).

### [validation.dataloader]

Same fields as `training.dataloader`.

---

## [debug]

- `seed` (default: null)
  Base RNG seed (global or per-PP depending on config).

- `deterministic` (default: false)
  Enable deterministic algorithms (slower).

- `deterministic_warn_only` (default: false)
  Warn instead of error when deterministic ops are unavailable.

- `moe_force_load_balance` (default: false)
  Force MoE routing balance (debug only).
