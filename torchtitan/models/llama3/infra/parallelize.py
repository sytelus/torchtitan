# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Parallelization for Llama3
================================

This file applies PyTorch Distributed (PT-D) parallelisms and training optimizations
to the Llama model. It serves as the reference implementation for how to parallelize
transformer models in TorchTitan.

ORDER OF PARALLELIZATION:
-------------------------
The parallelization is applied in a specific order for correctness:

1. **Tensor Parallelism (TP)**: Applied first because it modifies module structure
   and wraps parameters with DTensor. Must happen before other wrappers.

2. **Activation Checkpointing (AC)**: Applied after TP but before FSDP.
   This wraps modules to recompute activations during backward, reducing memory.
   Must be before FSDP so checkpointed modules are wrapped once.

3. **torch.compile**: Applied after AC because:
   - Compile captures the AC-wrapped graph
   - Per-block compilation is more efficient than whole-model compilation
   - Must be before FSDP to compile the local (un-sharded) module

4. **FSDP/DDP**: Applied last because it wraps the entire model.
   - FSDP shards parameters and requires all modifications to be complete
   - DDP wraps for gradient synchronization

TENSOR PARALLELISM EXPLAINED:
-----------------------------
For each transformer block, we apply TP to split computations across GPUs:

    Input: [batch, seq, hidden]
           |
    SequenceParallel(attention_norm)  # Keep seq sharded
           |
    PrepareModuleInput(Shard(1) -> Replicate)  # Gather seq for attention
           |
    ColwiseParallel(wq, wk, wv)  # Split heads across TP ranks
           |
    Attention computation (local)
           |
    RowwiseParallel(wo)  # Combine partial outputs, output Shard(1)
           |
    SequenceParallel(ffn_norm)
           |
    ColwiseParallel(w1, w3)  # Split FFN intermediate
           |
    RowwiseParallel(w2)  # Combine, output Shard(1)
           |
    Output: [batch, seq, hidden] (sharded on seq)

SEQUENCE PARALLELISM:
---------------------
Between TP operations, activations are sharded on the sequence dimension.
This reduces memory without extra communication (the existing TP comms
naturally transform between Replicate and Shard(seq)).

FSDP2 EXPLAINED:
----------------
FSDP (Fully Sharded Data Parallel) shards parameters across ranks:

    Before forward:
    - Parameters: sharded (each rank has 1/N of params)

    During forward (for each FSDP unit):
    - All-gather: Collect full params from all ranks
    - Compute forward with full params
    - Optionally reshard (free the gathered params)

    During backward:
    - All-gather params if resharded
    - Compute gradients
    - Reduce-scatter: Each rank gets its shard of gradients

KEY CONFIGURATION OPTIONS:
--------------------------
- reshard_after_forward="default": Don't reshard for PP (avoids per-microbatch all-gathers)
- cpu_offload=True: Move params/grads/optimizer to CPU (slower but less GPU memory)
- mp_policy: Controls param/reduce dtypes for mixed precision

GOTCHAS:
--------
1. seq_len must be divisible by (tp_degree * 2 * cp_degree) for proper sharding
2. TP with FP8 requires special parallel styles (Float8ColwiseParallel, etc.)
3. For PP, don't reshard after forward (expensive per-microbatch all-gathers)
4. The model must be on meta device for efficient parallelization
5. loss_parallel shards the vocab dimension for memory savings
"""

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.tools.logging import logger


# =============================================================================
# SELECTIVE OP ACTIVATION CHECKPOINTING
# These operations are expensive (compute or memory) and should be saved
# rather than recomputed during backward pass in selective AC mode.
# =============================================================================
_op_sac_save_list = {
    # Matrix multiplications - expensive to recompute
    torch.ops.aten.mm.default,
    # Attention ops - these are the most expensive operations
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    # Communication ops - must preserve for correct distributed behavior
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # For low precision training (FP8), always save max values since
    # the absolute maximum is used to compute scaling factors.
    # Recomputing could give slightly different results causing NaN.
    torch.ops.aten.max.default,
    # Higher-order ops that shouldn't be recomputed
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn.default,
    torch._higher_order_ops.inductor_compiled_code,
}


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    This is the main entry point for model parallelization. It orchestrates
    the application of all parallelism strategies in the correct order:
    TP -> AC -> compile -> FSDP/DDP

    Args:
        model: The model to parallelize. Should be on meta device for efficiency.
        parallel_dims: Configuration for all parallelism dimensions.
        job_config: Complete job configuration with training settings.

    Returns:
        nn.Module: The parallelized model, ready for distributed training.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory. Using meta device allows the
    parallelization to set up DTensor wrappers without allocating memory.
    """
    # =========================================================================
    # SEQUENCE LENGTH VALIDATION
    # For TP + CP, sequence length must be evenly divisible for proper sharding.
    # seq_len_divisor = tp * (cp * 2) because:
    # - TP splits heads, requiring even division
    # - CP splits sequence, with load balancing requiring factor of 2
    # =========================================================================
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # =========================================================================
    # STEP 1: TENSOR PARALLELISM
    # Applied first because it modifies the module structure by wrapping
    # parameters with DTensor. Other wrappers (AC, FSDP) work with this structure.
    # =========================================================================
    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(job_config, tp_mesh)

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # =========================================================================
    # STEP 2: ACTIVATION CHECKPOINTING
    # Applied after TP but before FSDP/compile for correct wrapping order.
    # AC reduces memory by recomputing activations during backward instead
    # of storing them. The _op_sac_save_list specifies expensive ops to save.
    # =========================================================================
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            # pyrefly: ignore [bad-argument-type]
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )
        # AC is applied before FSDP so that checkpointed modules are wrapped once.
        # If applied after FSDP, each shard would be wrapped separately, which
        # would be incorrect.

    # =========================================================================
    # STEP 3: TORCH.COMPILE
    # Applied per-TransformerBlock for efficiency. Whole-model compilation is
    # slower and less effective due to the repeated structure not being exploited.
    # Must be before FSDP because compile works on the local (un-sharded) module.
    # =========================================================================
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # =========================================================================
    # STEP 4: DATA PARALLELISM (FSDP or DDP)
    # Applied last because it wraps the entire model for distributed training.
    # FSDP shards parameters; DDP replicates and syncs gradients.
    # =========================================================================
    if parallel_dims.fsdp_enabled:
        # dp_mesh is the mesh for FSDP/HSDP
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        if parallel_dims.world_size != dp_replicate_mesh.size():
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_replicate_mesh,
            enable_compile=model_compile_enabled,
        )

    return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            # NOTE: when the fourth argument (positions) is not None, its input layout
            # and desired input layout should be Replicate()
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(
            transformer_block, backend=compile_config.backend, fullgraph=True
        )
        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        # pyrefly: ignore [bad-typed-dict-key]
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    fully_shard(model, **fsdp_config)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    # pyrefly: ignore [invalid-param-spec]
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
