# Architecture and Code Tour

This document is a high-level guide to how TorchTitan is structured and how a
training run flows through the codebase. It is written for readers who know
PyTorch but are new to multi-dimensional parallelism, FSDP, and FP8.

## Mental model

TorchTitan is a composable training harness:

1. You provide a config (TOML + CLI overrides).
2. A `TrainSpec` wires in model- and task-specific components.
3. The trainer builds the tokenizer, dataloader, model, and optimizer.
4. Parallelism, checkpointing, precision, and profiling are applied.
5. The training loop runs with optional validation and fault tolerance.

The key principle is *separation of concerns*: model code is mostly standard
PyTorch; distributed behavior is injected by adapters and wrappers.

## Repository map (where to start)

- `torchtitan/train.py`:
  The training entrypoint and main loop. This is the best place to understand
  the overall lifecycle and how config drives behavior.

- `torchtitan/config/`:
  `job_config.py` defines all configuration fields; `manager.py` parses TOML
  and CLI and merges them into a `JobConfig`.

- `torchtitan/protocols/train_spec.py`:
  The protocol for model-specific implementations. Each model registers a
  `TrainSpec` that provides builders for tokenizer, dataloader, model args,
  optimizers, and validation.

- `torchtitan/distributed/`:
  Parallelism building blocks and utilities: device meshes, FSDP/HSDP, TP,
  PP, context parallelism, expert parallelism.

- `torchtitan/components/`:
  Reusable training components (checkpointing, metrics, quantization, FT, etc.).

- `torchtitan/models/<model>/`:
  Model definition, model args, and model-specific infra (e.g., parallelization
  plans for Llama). Train configs live under `train_configs/`.

## Example: GPT-2 experiment layout

For a concrete example of the experiments layout, the GPT-2 tutorial lives in
`torchtitan/experiments/gpt2/`:

```
torchtitan/
├── models/              # Production models (Llama3, DeepSeek, etc.)
├── experiments/         # Experimental models (including GPT-2)
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

## Key concepts for model integrations

1. **TrainSpec**: Bundles model-specific components (model class, parallelization,
   optimizer, dataloader, etc.) so the generic trainer can work with any model.

2. **Model Protocol**: All models must implement `init_weights()` and accept
   `model_args` in `__init__()`.

3. **Parallelization Order**: TP → AC → compile → FSDP/DDP (order matters!)
   - **GPT-2 exception**: For fused forward+loss, GPT-2 compiles *after* FSDP/DDP.
     See `docs/tutorial_gpt2.md` for details.

## Parallelization order (TP → AC → compile → FSDP/DDP)

The order in which parallelization techniques are applied is critical for
correctness and performance. Each step must occur in sequence:

```
Original Model
     │
     ▼
[1] TP: Shard tensors across devices (establishes layouts)
     │
     ▼
[2] AC: Wrap blocks in CheckpointWrapper (respects TP layouts)
     │
     ▼
[3] Compile: Trace graph with AC visible (before FSDP hooks)
     │
     ▼
[4] FSDP/DDP: Add communication hooks (final step, after graph traced)
     │
     ▼
Build Optimizer (sees final sharded parameters)
```

**Step 1: Tensor Parallelism (TP) — First**

TP splits tensors across devices (e.g., column-wise for embeddings, row-wise for
attention outputs). It must be applied first because it establishes the tensor
sharding layouts that all subsequent steps must respect.

```python
# Example: embeddings split column-wise across TP ranks
"tok_embeddings": RowwiseParallel(
    input_layouts=Replicate(),
    output_layouts=Shard(1),
)
```

**Step 2: Activation Checkpointing (AC) — Second**

AC wraps transformer blocks in `CheckpointWrapper` to trade compute for memory
(recompute activations during backward instead of storing them). It must:
- Come **after TP** to respect the established tensor layouts
- Come **before compile** so the compiler can see and optimize the checkpointing pattern

**Step 3: torch.compile — Third**

Compile traces the computation graph and generates optimized kernels. It must:
- Come **after AC** to see the CheckpointWrapper and properly trace the control flow
- Come **before FSDP** because FSDP adds communication hooks (all-gather,
  reduce-scatter) that cause **graph breaks**

```python
# From parallelize.py - the comment explains the ordering
# turn on per-TransformerBlock compile after AC wrapping and before FSDP
if model_compile_enabled:
    apply_compile(model, job_config.compile)
```

**Step 4: FSDP/DDP — Last**

Data parallelism shards parameters across devices. Applied last because:
- FSDP adds hooks that would break the torch.compile graph if applied earlier
- The optimizer is built **after** parallelism, so it sees the final sharded parameters

**The Graph-Breaking Problem**

The critical constraint is avoiding graph breaks for `torch.compile`. If FSDP is
applied before compile, the communication hooks fragment the graph, forcing
activation checkpointing to fall back to eager execution — destroying performance.

## Training flow (what happens in `train.py`)

1. **Config parsing**
   - `ConfigManager.parse_args()` loads TOML, merges CLI overrides, validates,
     and returns a `JobConfig`.

2. **Distributed init**
   - `dist_utils.init_distributed()` sets up process groups and flight recorder
     environment variables, then `ParallelDims` builds device meshes.

3. **Tokenizer + dataloader**
   - `TrainSpec.build_tokenizer_fn()` and `build_dataloader_fn()` are called to
     create dataset and loading logic.

4. **Model construction (meta device)**
   - The model is created on the meta device to avoid immediate memory
     allocation. This enables large models and FSDP-friendly initialization.

5. **Model converters (optional)**
   - Converters (e.g., Float8) swap modules or wrap layers before parallelism.

6. **Apply parallelism + initialize weights**
   - For PP: the model is split into stages and each stage is parallelized.
   - For non-PP: apply TP + AC + compile + FSDP/HSDP/DDP.
   - Weights are initialized *after* sharding via `to_empty()` and
     `init_weights()`.

7. **Optimizer + scheduler**
   - Built after parallelism so sharded parameters are visible to the optimizer.

8. **Checkpointing + metrics**
   - `CheckpointManager` wires model/optimizer/stateful objects for DCP.
   - `MetricsProcessor` tracks throughput, loss, memory, MFU, etc.

9. **Training loop**
   - Gradient accumulation, loss rescaling, and distributed loss reduction.
   - Optional validation and profiling.

## How TrainSpec decouples model-specific behavior

A `TrainSpec` provides the *contract* between the trainer and a particular
model. It is responsible for:

- model args (and how they are derived from `JobConfig`)
- tokenizer and dataloader construction
- loss function and validation
- how to parallelize the model
- how to build optimizers/schedulers

This allows `train.py` to stay generic while models live in their own packages.

## Where to customize or extend

- Add a new model: follow `torchtitan/models/README.md`.
- Override config structure: use `job.custom_config_module`.
- Replace components (metrics, checkpointing): implement new builders in a
  custom TrainSpec or extend existing ones.

## Next reading

- `docs/parallelism_primer.md` for the parallelism + mesh model.
- `docs/config_reference.md` for a full option list and defaults.
- `docs/fsdp.md` and `docs/float8.md` for deeper dives into FSDP2 and FP8.
