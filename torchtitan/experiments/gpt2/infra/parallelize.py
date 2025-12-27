"""
GPT-2 Model Parallelization
===========================

This module applies PyTorch Distributed parallelism to the GPT-2 model.
For simplicity, this implementation supports:
- Single GPU (no parallelism)
- DDP (Distributed Data Parallel)
- FSDP (Fully Sharded Data Parallel)

Tensor Parallelism (TP) is not implemented for GPT-2 in this tutorial,
but the structure follows TorchTitan conventions for easy extension.

ORDER OF PARALLELIZATION (GPT-2):
1. Tensor Parallelism (TP) - if enabled (not implemented here)
2. Activation Checkpointing (AC) - reduces memory
3. FSDP/DDP - data parallelism
4. Weight tying (if enabled)
5. torch.compile - whole-model optimization

Note: TorchTitan's standard order for large models is TP -> AC -> compile -> FSDP/DDP.
GPT-2 intentionally compiles AFTER data parallelism so the compiled graph can
fuse the output projection with cross-entropy loss.

COMPILATION STRATEGY:
---------------------
This module uses WHOLE-MODEL compilation (including loss) after data parallelism
because it can be superior for small models:

1. FUSED OUTPUT + LOSS: Wrapping output projection + cross_entropy in a single
   torch.compile call enables fusion that avoids materializing the full
   [batch, seq, vocab_size] logits tensor. For GPT-2 with vocab=50304:
   - Logits: 64 × 1024 × 50304 × 2 bytes ≈ 6.6GB in bf16
   - Fused: Only chunked computation, ~50% memory reduction
   (See: https://github.com/pytorch/torchtune/pull/2507)

2. PER-LAYER vs WHOLE-MODEL: PyTorch docs note "minimal speedup differences"
   between regional and full-model compilation for runtime. Per-layer mainly
   reduces COMPILE TIME (67s → 9.6s), not execution time.
   (See: https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)

3. SMALL MODELS: For models with few layers (e.g., 6-layer tiny), whole-model
   compilation may produce better code through cross-layer fusion.

This implementation compiles the entire model (including loss) so torch.compile
can fuse output projection + cross-entropy and avoid materializing full logits.

CHOOSING BETWEEN DDP AND FSDP:
- DDP (data_parallel_shard_degree=1, data_parallel_replicate_degree=N):
  Each GPU holds a full model replica. Gradients are synchronized via
  all-reduce. Best for small models that fit in GPU memory, as it avoids
  the communication overhead of parameter sharding.

- FSDP (data_parallel_shard_degree=N):
  Parameters are sharded across GPUs. Each GPU only holds 1/N of the
  parameters. Before each forward/backward, parameters are all-gathered;
  after backward, gradients are reduce-scattered. Best for large models
  that don't fit on a single GPU.

FOR SMALL MODELS ON HIGH-MEMORY GPUS (e.g., GPT-2 124M on B200 with 192GB):
  DDP is strongly preferred because:
  1. No all-gather/reduce-scatter overhead for parameters
  2. Only gradient all-reduce is needed (same as FSDP)
  3. Simpler memory access patterns, better cache utilization
  4. GPT-2 124M (~500MB in bf16) easily fits on B200's 192GB memory
  FSDP overhead would dominate for such small models.
"""

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.tools.logging import logger


# Operations to SAVE (not recompute) during selective activation checkpointing.
#
# Activation checkpointing trades compute for memory: instead of storing all
# intermediate activations for the backward pass, we discard them and recompute
# during backward. Selective AC optimizes this by saving outputs of expensive
# operations while recomputing cheap ones.
#
# SAVE these operations (expensive to recompute):
#   - mm (matrix multiply): O(n³) compute, dominates transformer cost
#   - scaled_dot_product_attention variants: O(n²) attention with optimized
#     CUDA kernels (Flash Attention, etc.) that are expensive to re-execute
#
# RECOMPUTE these operations (cheap, not in this list):
#   - Element-wise ops (GELU, dropout, add): O(n) compute, memory-bound
#   - LayerNorm: O(n) compute, negligible vs matmul/attention
#   - Reshape/transpose: Zero compute, just metadata
#
# The savings come from not storing large activation tensors (e.g., attention
# scores of shape [batch, heads, seq, seq]) while paying minimal recompute cost
# for cheap operations.
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}


def parallelize_gpt2(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """
    Apply parallelism strategies to GPT-2 model.

    This function applies parallelism in the correct order:
    AC -> FSDP/DDP -> weight tying -> compile

    Args:
        model: The GPT-2 model (preferably on meta device)
        parallel_dims: Configuration for parallelism dimensions
        job_config: Complete job configuration

    Returns:
        The parallelized model ready for distributed training
    """
    # Check for unsupported parallelism modes
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor Parallelism is not implemented for GPT-2 in this tutorial. "
            "Set tensor_parallel_degree=1 in your config."
        )

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline Parallelism is not implemented for GPT-2 in this tutorial. "
            "Set pipeline_parallel_degree=1 in your config."
        )

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallelism is not implemented for GPT-2 in this tutorial. "
            "Set context_parallel_degree=1 in your config."
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # =========================================================================
    # STEP 1: ACTIVATION CHECKPOINTING
    # Reduces memory by recomputing activations during backward pass
    # =========================================================================
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )
        logger.info("Applied Activation Checkpointing to the model")

    # =========================================================================
    # STEP 2: DATA PARALLELISM (FSDP or DDP)
    # NOTE: For GPT-2, we apply data parallelism BEFORE compilation.
    # This is different from TorchTitan's standard order (compile before DP).
    # Reason: We want to compile the WHOLE model (including loss computation)
    # in one graph, which requires the model to be fully constructed first.
    # =========================================================================
    if parallel_dims.fsdp_enabled:
        # Use FSDP for sharded data parallelism
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP (DDP + FSDP) to the model")
        else:
            logger.info("Applied FSDP to the model")

    elif parallel_dims.dp_replicate_enabled:
        # Use DDP for replicated data parallelism (no sharding)
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        if parallel_dims.world_size != dp_replicate_mesh.size():
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_replicate_mesh,
            enable_compile=model_compile_enabled,
        )
    else:
        logger.info("No data parallelism applied (single GPU mode)")

    # =========================================================================
    # STEP 3: WEIGHT TYING (AFTER FSDP/DDP, BEFORE COMPILE)
    # Weight tying must be applied AFTER FSDP/DDP to avoid sharding issues.
    # If applied before FSDP, the shared parameter would be processed twice:
    # - Once when sharding tok_embeddings
    # - Once when sharding output
    # This could cause incorrect gradient computation or memory issues.
    #
    # Weight tying must be applied BEFORE torch.compile because GPT-2 uses
    # whole-model compilation. If applied after compile, the compiled graph
    # would have captured the original output.weight tensor, not the tied one.
    #
    # By applying weight tying after FSDP but before compile:
    # - Both tok_embeddings.weight and output.weight are already DTensors
    # - Assigning one to the other makes them share the same DTensor
    # - The compiled graph will see the tied weights from the start
    # - The optimizer (built after this) sees only one copy of the parameter
    #
    # NOTE: Qwen3 uses per-layer compilation (which doesn't compile output),
    # so it applies weight tying after compile. GPT-2 uses whole-model
    # compilation, so weight tying must happen before compile.
    # =========================================================================
    if model.model_args.weight_tying:
        model.output.weight = model.tok_embeddings.weight
        logger.info(
            "Applied weight tying: output.weight now shares tok_embeddings.weight"
        )

    # =========================================================================
    # STEP 4: TORCH.COMPILE (WHOLE MODEL)
    # Compiles the entire model including forward + loss computation.
    # This enables fusion of output projection with cross-entropy loss.
    # Must happen AFTER weight tying so the compiled graph sees tied weights.
    # =========================================================================
    if model_compile_enabled:
        model = apply_compile(model, job_config.compile)

    return model


def apply_compile(model: nn.Module, compile_config: CompileConfig) -> nn.Module:
    """
    Apply torch.compile to the ENTIRE model (whole-model compilation).

    GPT-2 uses whole-model compilation instead of per-layer compilation because:

    1. FUSED OUTPUT + LOSS: The GPT-2 model computes loss inside forward() when
       labels are provided. Whole-model compilation allows torch.compile to fuse
       the output projection (model.output) with cross-entropy loss, avoiding
       materialization of the full [batch, seq, vocab_size] logits tensor.

       Memory savings for GPT-2 (vocab=50304, batch=64, seq=1024):
       - Without fusion: ~6.6GB for logits tensor in bf16
       - With fusion: ~50% reduction through chunked computation

    2. SMALL MODEL BENEFITS: For small models like GPT-2 (4-48 layers), the
       "compile once, reuse N times" benefit of per-layer compilation is minimal.
       Whole-model compilation provides:
       - Cross-layer fusion opportunities
       - Better global memory planning
       - Simpler compilation (one graph vs N graphs)

    3. RESEARCH VALIDATION: PyTorch docs note "minimal speedup differences"
       between regional and full-model compilation for runtime. Per-layer mainly
       reduces COMPILE TIME, not execution time.
       (See: https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)

    TRADE-OFFS:
    - Longer initial compilation time (one large graph)
    - May hit graph size limits for very large models (use per-layer for those)
    - Less granular error messages on compilation failures

    WHY COMPILE AFTER DATA PARALLELISM:
    We apply DP (FSDP/DDP) before compilation so that:
    1. The model structure is finalized
    2. DDP/FSDP hooks are in place
    3. torch.compile can see and optimize around communication patterns

    Args:
        model: The GPT-2 model (already wrapped with FSDP/DDP if applicable)
        compile_config: Configuration for torch.compile (backend, etc.)

    Returns:
        The compiled model
    """
    # Compile the entire model as one unit.
    # This includes all layers, embeddings, norm, output projection, AND
    # the loss computation (which is inside forward when labels are provided).
    compiled_model = torch.compile(
        model,
        backend=compile_config.backend,
        # NOTE: We don't use fullgraph=True here because FSDP/DDP may introduce
        # graph breaks at communication boundaries. torch.compile will still
        # optimize each subgraph effectively.
    )

    logger.info(
        f"Compiled entire GPT-2 model with torch.compile (backend={compile_config.backend}). "
        "Forward+loss will be fused for optimal memory efficiency."
    )

    return compiled_model


# Threshold for FSDP sharding granularity (in number of parameters).
# Layers with fewer parameters than this should be grouped together to reduce
# communication overhead. Based on PyTorch FSDP best practices:
# - Layers with >100M parameters benefit most from individual sharding
# - Layers with <100K parameters should NOT be sharded separately
# - Small layers incur communication overhead that dominates compute time
#
# For GPT-2 models (all layers are well below 100M):
#   - tiny: ~1.8M params/layer → group aggressively
#   - 124M: ~10M params/layer → group 10 layers together
#
# See: https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html
_FSDP_MIN_PARAMS_PER_UNIT = 100_000_000  # 100M parameters


def _count_parameters(module: nn.Module) -> int:
    """Count total parameters in a module (excluding nested FSDP-wrapped modules)."""
    return sum(p.numel() for p in module.parameters())


def _estimate_model_memory_mb(model: nn.Module, dtype: torch.dtype) -> float:
    """Estimate model memory in MB for a given dtype."""
    total_params = _count_parameters(model)
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float8_e4m3fn: 1,
        torch.float8_e5m2: 1,
    }.get(dtype, 4)
    return (total_params * bytes_per_param) / (1024 * 1024)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """
    Apply FSDP2 (Fully Sharded Data Parallel) to the model with smart layer grouping.

    FSDP shards model parameters across GPUs to reduce per-GPU memory usage.
    For a model with P parameters and N GPUs, each GPU only stores P/N parameters.

    WHEN TO USE FSDP vs DDP:
    ========================
    FSDP adds communication overhead (all-gather before forward, reduce-scatter
    after backward) that DDP doesn't have. This overhead is only worthwhile when:
    1. Model doesn't fit in GPU memory (FSDP required)
    2. Model is large enough that sharding benefits outweigh overhead

    RECOMMENDATION FOR SMALL MODELS:
    For GPT-2 on high-memory GPUs (e.g., B200 with 192GB), DDP is strongly
    preferred. Even GPT-2 1.5B (~3GB in bf16) fits easily, and DDP avoids
    the all-gather/reduce-scatter overhead that FSDP incurs.

    Research shows FSDP can be 3x slower than DDP for small models due to
    communication overhead dominating compute time.

    SMART LAYER GROUPING STRATEGY:
    ==============================
    Per-layer sharding (one FSDP unit per TransformerBlock) is inefficient for
    small models because each FSDP unit incurs communication overhead:
    - All-gather before forward: O(params) communication
    - Reduce-scatter after backward: O(params) communication

    For layers with <100M parameters, the communication overhead can exceed
    the memory savings benefit. This implementation groups layers together
    to reduce the number of FSDP units while maintaining memory benefits.

    Grouping strategy:
    1. Calculate total parameters per layer
    2. Group consecutive layers until combined params approach 100M threshold
    3. Shard each group as a single FSDP unit

    Example for GPT-2 124M (12 layers × ~10M params/layer = ~120M total):
    - Without grouping: 12 FSDP units (12 all-gather + 12 reduce-scatter calls)
    - With grouping: 2 FSDP units of 6 layers each (~60M params/unit)
    - Result: 6x fewer communication operations

    SHARDING DETAILS:
    =================
    1. Embeddings (tok_embeddings, pos_embeddings):
       - Sharded with reshard_after_forward=True
       - For GPT-2: tok_embeddings is large (vocab_size × dim = 50304 × 768 ≈ 155MB)

    2. Transformer blocks (grouped for efficiency):
       - Multiple layers grouped into single FSDP units
       - reshard_after_forward=True to free memory between groups

    3. Final layers (norm + output projection):
       - Sharded with reshard_after_forward=False (keep gathered after forward)
       - Backward starts immediately, so no benefit to resharding

    4. Root model wrapper:
       - fully_shard(model) creates root FSDP unit for remaining parameters

    Args:
        model: The model to shard
        dp_mesh: Device mesh for data parallelism
        param_dtype: Dtype for parameters (e.g., bfloat16 for memory savings)
        reduce_dtype: Dtype for gradient reductions (e.g., float32 for precision)
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # Calculate model size and warn if DDP would be more appropriate
    model_size_mb = _estimate_model_memory_mb(model, param_dtype)
    total_params = _count_parameters(model)
    num_layers = len(model.layers)

    if total_params < _FSDP_MIN_PARAMS_PER_UNIT:
        logger.warning(
            f"Model has {total_params:,} parameters ({model_size_mb:.1f}MB in {param_dtype}), "
            f"which is below the recommended FSDP threshold of {_FSDP_MIN_PARAMS_PER_UNIT:,}. "
            "DDP (data_parallel_shard_degree=1) is recommended for better performance. "
            "FSDP communication overhead may dominate compute time for small models."
        )

    # Shard embeddings - these are large (vocab_size × dim) and benefit from sharding.
    # reshard_after_forward=True frees memory after embeddings are computed.
    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    if model.pos_embeddings is not None:
        fully_shard(model.pos_embeddings, **fsdp_config, reshard_after_forward=True)

    # Smart layer grouping: group consecutive layers until we reach the threshold.
    # This reduces the number of FSDP units (and thus communication operations)
    # while maintaining the memory benefits of sharding.
    layer_items = list(model.layers.items())
    layers_per_group = 1  # Default: one layer per FSDP unit
    num_fsdp_units = 0

    if num_layers > 0:
        # Calculate average parameters per layer
        sample_layer = next(iter(model.layers.values()))
        params_per_layer = _count_parameters(sample_layer)

        # Determine optimal group size based on the 100M threshold
        # We want each FSDP unit to have ~100M parameters for optimal efficiency
        if params_per_layer >= _FSDP_MIN_PARAMS_PER_UNIT:
            # Large layers: shard each layer individually (standard approach)
            layers_per_group = 1
        else:
            # Small layers: group to reduce communication overhead
            # Target: combined params per group ≈ _FSDP_MIN_PARAMS_PER_UNIT
            layers_per_group = max(1, _FSDP_MIN_PARAMS_PER_UNIT // params_per_layer)
            # Don't create groups larger than the model
            layers_per_group = min(layers_per_group, num_layers)

        logger.info(
            f"FSDP layer grouping: {num_layers} layers, {params_per_layer:,} params/layer, "
            f"grouping {layers_per_group} layers per FSDP unit "
            f"(~{params_per_layer * layers_per_group:,} params/unit)"
        )

        # Group layers and shard each group
        for i in range(0, num_layers, layers_per_group):
            group_end = min(i + layers_per_group, num_layers)
            group_layers = [layer_items[j][1] for j in range(i, group_end)]
            num_fsdp_units += 1

            if len(group_layers) == 1:
                # Single layer: shard directly
                fully_shard(group_layers[0], **fsdp_config, reshard_after_forward=True)
            else:
                # Multiple layers: shard as a group
                # Note: fully_shard accepts a list of modules to group together
                # pyrefly: ignore [no-matching-overload]
                fully_shard(group_layers, **fsdp_config, reshard_after_forward=True)

    # Shard final layers but DON'T reshard after forward.
    # Optimization: backward pass starts immediately after forward ends, so these
    # parameters would be all-gathered again right away. Keeping them gathered
    # avoids a redundant reshard/all-gather cycle.
    if model.norm is not None and model.output is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard([model.norm, model.output], **fsdp_config, reshard_after_forward=False)

    # Wrap the entire model as the root FSDP unit.
    # This handles any remaining parameters and provides the top-level FSDP context.
    fully_shard(model, **fsdp_config)

    logger.info(
        f"Applied FSDP2 to the model with smart layer grouping "
        f"({num_layers} layers → {num_fsdp_units} FSDP units)"
    )


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    """
    Apply DDP (Distributed Data Parallel) using the composable replicate() API.

    WHAT replicate() DOES:
    1. Registers gradient hooks on all parameters
    2. During backward, gradients are synchronized across GPUs via all-reduce
    3. After backward, all GPUs have identical averaged gradients
    4. Optimizer step updates parameters identically on all GPUs

    Unlike FSDP, DDP keeps full model replicas on each GPU - no parameter sharding.
    Each GPU processes different data batches, but model weights stay synchronized.

    WHY DDP FOR SMALL MODELS:
    - No all-gather overhead before forward (parameters already local)
    - No reduce-scatter overhead after backward (only all-reduce on gradients)
    - Simpler memory access patterns, better GPU cache utilization
    - For GPT-2 124M (~500MB in bf16), DDP is faster than FSDP on modern GPUs

    bucket_cap_mb=100: Gradients are bucketed into 100MB chunks for all-reduce.
    Larger buckets = fewer all-reduce calls = better bandwidth utilization.
    Trade-off: larger buckets delay gradient sync, reducing overlap with backward.

    LOGITS AND LOSS FLOW:
    - Each GPU runs forward on its local batch, producing local logits
    - Loss is computed locally on each GPU's batch portion
    - Gradients flow backward through the local model copy
    - All-reduce averages gradients across all GPUs
    - Result: each GPU has the same gradients as if trained on the full batch

    Args:
        model: The model to replicate
        dp_mesh: Device mesh specifying which GPUs to replicate across
        enable_compile: If True, enable DDP-specific optimizations for torch.compile
    """
    if enable_compile:
        # Enable DDP-specific optimizations in torch.compile:
        # - Fuses gradient bucketing with backward graph
        # - Reduces CPU overhead of gradient synchronization
        # - Enables better overlap of communication and computation
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    # replicate() is the composable API equivalent of torch.nn.parallel.DistributedDataParallel.
    # It modifies the model in-place to add gradient synchronization hooks.
    # Unlike the legacy DDP wrapper, it doesn't change the model's type or interface.
    # pyrefly: ignore [invalid-param-spec]
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
