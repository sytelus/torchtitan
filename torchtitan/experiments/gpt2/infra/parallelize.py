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

ORDER OF PARALLELIZATION:
1. Tensor Parallelism (TP) - if enabled (not implemented here)
2. Activation Checkpointing (AC) - reduces memory
3. torch.compile - optimizes computation
4. FSDP/DDP - data parallelism
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


# Operations to save (not recompute) during selective activation checkpointing
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
    AC -> compile -> FSDP/DDP

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
    # STEP 2: TORCH.COMPILE
    # Compiles each transformer block for better performance
    # =========================================================================
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # =========================================================================
    # STEP 3: DATA PARALLELISM (FSDP or DDP)
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

    return model


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """Apply torch.compile to each transformer block."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(
            transformer_block, backend=compile_config.backend, fullgraph=True
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """Apply FSDP to the model."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # Shard embeddings
    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    if model.pos_embeddings is not None:
        fully_shard(model.pos_embeddings, **fsdp_config, reshard_after_forward=True)

    # Shard each transformer block
    for layer_id, transformer_block in model.layers.items():
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=True)

    # Shard final layers (don't reshard - they're used immediately)
    if model.norm is not None and model.output is not None:
        fully_shard([model.norm, model.output], **fsdp_config, reshard_after_forward=False)

    # Shard the entire model
    fully_shard(model, **fsdp_config)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    """Apply DDP to the model."""
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
