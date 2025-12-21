# How to use checkpointing in `torchtitan`

You may want to enable checkpointing in `torchtitan` for better fault tolerance during training, or to enable easier importing and exporting of weights between `torchtitan` and other libraries. `torchtitan` offers varying degrees of support for other checkpoint formats which are listed further below.

## Async Checkpointing

Standard synchronous checkpointing blocks training while saving, which can take
minutes for large models. TorchTitan offers three async modes to reduce this
overhead.

### The Three Modes

| Mode | How it Works | Blocking Time | Memory Cost |
|------|--------------|---------------|-------------|
| `disabled` | Synchronous save | High (minutes for large models) | None |
| `async` | Background threads via `dcp.async_save` | Low (tens of seconds) | GPU memory for staging |
| `async_with_pinned_mem` | Separate process + pinned CPU memory | Near-zero (<1s) | High CPU memory |

### Why `disabled` is the Default

The conservative default ensures safety across all environments:

1. **CPU Memory Safety**: The `async_with_pinned_mem` mode requires significant
   CPU memory for pinned buffers that persist between checkpoints. From PyTorch
   docs: pinned memory uses page-locked memory which "can be scarce as compared
   to pageable memory."

2. **GIL Contention**: Async modes use background threads that compete for
   Python's Global Interpreter Lock (GIL), which can cause CPU stalls and
   temporarily reduce training throughput during checkpoint writes.

3. **Memory Multiplication**: Async checkpointing copies model state to CPU
   buffers, effectively multiplying memory requirements by
   `checkpoint_size_per_rank × number_of_ranks`.

4. **Simplicity**: Synchronous checkpointing is predictable—training blocks
   until save completes, making debugging easier.

### When to Enable Each Mode

```toml
# For debugging/development (simplest, most predictable)
[checkpoint]
async_mode = "disabled"

# For most production training (good balance of speed vs memory)
[checkpoint]
async_mode = "async"

# For maximum throughput (requires ample CPU memory)
[checkpoint]
async_mode = "async_with_pinned_mem"
```

**Recommendation from TorchTitan source code**: "Use `async_with_pinned_mem` for
production training (near-zero overhead)" — but only if you have sufficient CPU
memory.

### Performance Characteristics

At scale (1856 GPUs training Llama3-70B), async checkpointing with cached plans
reduced background processing time from ~436 seconds to ~67 seconds (6.5x
improvement). For the Llama 3.1 8B model, TorchTitan achieves 5-15x reduction in
checkpointing overhead compared to synchronous distributed checkpointing.

### Considerations

1. **FSDP CPU Offload Conflict**: If using `training.enable_cpu_offload=true`,
   be cautious with `async_with_pinned_mem` as both compete for CPU memory.

2. **Checkpoint Frequency**: If checkpointing every 1000+ steps, synchronous
   save overhead may be negligible compared to total training time.

3. **Large Models**: For very large models (70B+), async checkpointing becomes
   more important as synchronous saves can take many minutes.

4. **Pinned Memory Persistence**: With `async_with_pinned_mem`, the staging
   buffer is maintained between checkpoints, causing sustained memory pressure
   throughout training (unlike `async` mode where buffers are released after
   each save).

---

## A general guide to use checkpoints during training

1. ENABLE CHECKPOINTING
In your `torchtitan` training config, ensure that under `[checkpoint]`, `enable` is set to True.
```
[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
```
2. SAVE MODEL ONLY
By setting `last_save_model_only` to `True`, the checkpoint will only contain the model and exclude the optimizer state and extra train states, resulting in a smaller checkpoint size.
```
[checkpoint]
enable = true
last_save_model_only = true
```

3. CHOOSE DESIRED EXPORT PRECISION
The default model states are in `float32`. You can choose to export the checkpoint in a lower precision format such as `bfloat16`.
```
[checkpoint]
enable = true
last_save_model_only = true
export_dtype = "bfloat16"
```

4. EXCLUDING SPECIFIC KEYS FROM CHECKPOINT LOADING
In some cases, you may want to partially load from a previous-trained checkpoint and modify certain settings, such as the number of GPUs or the current step. To achieve this, you can use the `exclude_from_loading` parameter to specify which keys should be excluded from loading.
This parameter takes a list of string that should be excluded from loading.
```
[checkpoint]
enable = true
exclude_from_loading = ["data_loader", "lr_scheduler"]
```
When used in command line, the parameter should be a comma-separated list of strings. For example: `--checkpoint.exclude_from_loading data_loader,lr_scheduler`.

5. EXAMPLE CHECKPOINT CONFIGURATION
```
[checkpoint]
enable = true
folder = "checkpoint"
interval = 10
load_step = 5
last_save_model_only = true
export_dtype = "bfloat16"
```

A more exhaustive and up-to-date list of checkpoint config options can be found in `torchtitan/config/job_config.py`

## Creating a seed checkpoint
Sometimes one needs to create a seed checkpoint to initialize a model from step 0.
E.g. it is hard, if not impossible, for meta initialization on multiple devices to reproduce the initialization on a single device.
A seed checkpoint does initialization of the model on a single CPU, and can be loaded from another job on an arbitrary number of GPUs via DCP resharding.

To create a seed checkpoint, use the same model config as you use for training.
e.g.
```bash
NGPU=1 CONFIG_FILE=<path_to_model_config> ./run_train.sh --checkpoint.enable --checkpoint.create_seed_checkpoint --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 1 --parallelism.pipeline_parallel_degree 1 --parallelism.context_parallel_degree 1 --parallelism.expert_parallel_degree 1
```

## Conversion support

### HuggingFace
`torchtitan` offers two ways to work with Hugging Face models: either by directly saving and loading a Hugging Face checkpoint during training, or by using an example conversion script to directly reformat the model weights on cpu.

1. You can directly save huggingface model weights during training by using the `--checkpoint.last_save_in_hf` and `--checkpoint.last_save_model_only` options together. To directly load a `torchtitan` training session from a huggingface safetensors file, enable `--checkpoint.initial_load_in_hf`, and set either `--model.hf_assets_path` or `--checkpoint.initial_load_path` to the directory containing the huggingface checkpoint. `--checkpoint.initial_load_path` overrides `--model.hf_assets_path` if both are set.

2. To directly reformat the weights without the need to run a training loop, run the corresponding conversion script. The naming scheme is `torchtitan`-centric, e.g. convert_from_hf means convert hf->tt.

```bash
python ./scripts/checkpoint_conversion/convert_from_hf.py <input_dir> <output_dir> --model_name <model_name> --model_flavor <model_flavor>
python ./scripts/checkpoint_conversion/convert_to_hf.py <input_dir> <output_dir> --hf_assets_path ./assets/hf/Llama3.1-8B --model_name <model_name> --model_flavor <model_flavor>
# e.g.
python ./scripts/convert_from_hf.py ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/ ./initial_load_path/ --model_name llama3 --model_flavor 8B
```

### Torch

This guide will walk you through the steps required to convert a checkpoint from `torchtitan` so that it can be loaded into pt format.

1. CHECKPOINT CONFIGURATION
```
[checkpoint]
enable = true
folder = "checkpoint"
interval = 10
last_save_model_only = true
export_dtype = "bfloat16"
```

2. SAVE THE FINAL CHECKPOINT\
Once the above have been set, the final checkpoint at the end of the training step will consist of model only with the desired export dtype. However, if the final step has not been reached yet, full checkpoints will still be saved so that training can be resumed.

3. CONVERT SHARDED CHECKPOINTS TO A SINGLE FILE\
Finally, once you have obtained the last checkpoint, you can use the following command to convert the sharded checkpoints to a single .pt file.

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```


That's it. You have now successfully converted a sharded `torchtitan` checkpoint for use with pytorch formats.
