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
