# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan Training Loop - Core Training Infrastructure
=======================================================

This module implements the core training loop for distributed large language model (LLM)
training using PyTorch's native distributed training capabilities. It orchestrates
multiple parallelism strategies simultaneously:

PARALLELISM STRATEGIES OVERVIEW:
--------------------------------
1. **Data Parallelism (DP)**: Replicates model across GPUs, each processes different data.
   - FSDP (Fully Sharded Data Parallel): Shards model parameters across GPUs to save memory.
     Instead of each GPU holding the full model, each holds only a shard (1/N of parameters).
     Before each forward/backward, parameters are gathered (all-gather), then resharded
     (reduce-scatter) for gradients. This trades communication for memory.
   - DDP (Distributed Data Parallel): Full model replica on each GPU, gradients averaged.
   - HSDP (Hybrid Sharded Data Parallel): Combines FSDP within node + DDP across nodes.

2. **Tensor Parallelism (TP)**: Splits individual layers across GPUs.
   - Each GPU holds only a portion of weight matrices (column-wise or row-wise split).
   - Requires communication during each layer's forward/backward pass.
   - Best for very wide layers (large hidden dimensions).
   - See `tensor_parallel.py` for implementation details.

3. **Pipeline Parallelism (PP)**: Splits model layers across GPUs.
   - Different GPUs hold different transformer blocks (e.g., GPU0: layers 0-7, GPU1: layers 8-15).
   - Uses micro-batching to overlap computation and communication.
   - Schedule determines how micro-batches flow through pipeline (e.g., 1F1B, Interleaved1F1B).
   - See `pipeline_parallel.py` for implementation details.

4. **Context Parallelism (CP)**: Splits sequence dimension across GPUs.
   - Each GPU processes a portion of the input sequence.
   - Useful for very long sequences that don't fit in single GPU memory.
   - Requires attention to handle cross-sequence-chunk attention.

5. **Expert Parallelism (EP)**: For Mixture-of-Experts (MoE) models.
   - Different experts are placed on different GPUs.
   - Tokens are routed to appropriate expert GPUs via all-to-all communication.

MEMORY OPTIMIZATION TECHNIQUES:
-------------------------------
- **Activation Checkpointing (AC)**: Trades compute for memory by recomputing activations
  during backward pass instead of storing them. Can be applied per-layer or per-operation.
- **Mixed Precision Training**: Uses FP16/BF16 for forward/backward, FP32 for master weights.
  Reduces memory footprint and enables tensor cores for faster computation.
- **FP8 Training**: Experimental 8-bit training for further memory/compute savings.
  Requires SM89+ hardware (H100) and torchao library.
- **CPU Offloading**: Moves parameters/gradients/optimizer states to CPU when not in use.

KEY ABSTRACTIONS:
-----------------
- `TrainSpec`: Bundles model-specific configurations (model class, parallelization function,
  optimizer builder, etc.). Allows same training loop for different architectures.
- `ParallelDims`: Manages the multi-dimensional device mesh for all parallelism strategies.
- `JobConfig`: Hierarchical configuration system for all training hyperparameters.

TYPICAL TRAINING FLOW:
----------------------
1. Initialize distributed environment and device mesh
2. Build tokenizer and dataloader
3. Construct model on meta device (deferred initialization)
4. Apply model converters (e.g., FP8 linear layers)
5. Apply parallelization (TP, AC, compile, FSDP/DDP)
6. Materialize weights and initialize
7. Build optimizer and LR scheduler
8. Load checkpoint if available
9. Training loop: for each step, gradient accumulation, optimizer step, checkpointing

GOTCHAS AND TIPS:
-----------------
- Always ensure sequence_length is divisible by (TP_degree * 2 * CP_degree) for proper sharding.
- Pipeline parallel requires compatible schedule with number of stages and microbatches.
- When using async TP, torch.compile must be enabled for the model.
- FP8 training is experimental and requires specific hardware/software setup.
- Gradient clipping must consider PP to reduce gradients across stages.
"""

import dataclasses
import importlib
import json
import os
import time
from datetime import timedelta
from typing import Any, Iterable

import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    """
    Core training orchestrator for distributed LLM training.

    The Trainer class manages the complete training lifecycle including:
    - Distributed initialization and device mesh setup
    - Model construction and parallelization
    - Training loop execution with gradient accumulation
    - Checkpointing and recovery
    - Metrics logging and profiling

    The class implements `torch.distributed.checkpoint.stateful.Stateful` to enable
    checkpointing of training state (current step, tokens seen) alongside model/optimizer.

    EXTENDING THE TRAINER:
    ----------------------
    To support new model architectures, create a `TrainSpec` with model-specific:
    - Model class and args
    - Parallelization function (how to apply TP, AC, FSDP)
    - Pipelining function (for PP support)
    - Optimizer/LR scheduler builders

    The Trainer uses these via composition, allowing the same training loop for
    different architectures (Llama, DeepSeek, Flux, etc.).

    STATEFUL CHECKPOINTING:
    -----------------------
    The Trainer's `state_dict()` and `load_state_dict()` methods save/restore:
    - `step`: Current training step number
    - `ntokens_seen`: Total tokens processed (for logging/resumption)

    This allows seamless training resumption from checkpoints.

    Attributes:
        job_config (JobConfig): Complete training configuration (hyperparameters,
            parallelism settings, checkpoint config, etc.)
        parallel_dims (ParallelDims): Manages the multi-dimensional device mesh.
            Provides access to sub-meshes for each parallelism dimension (TP, PP, DP, etc.)
        train_spec (TrainSpec): Model-specific configuration bundle containing
            model class, parallelization functions, and component builders.

        tokenizer (BaseTokenizer | None): Tokenizer for text-to-token conversion.
            None for non-text models (e.g., image generation).
        dataloader (BaseDataLoader): Data loading iterator, supports checkpointing
            for resumption from exact data position.
        model_parts (list[nn.Module]): List of model chunks. Single element for non-PP,
            multiple elements for pipeline parallel (one per virtual stage on this rank).
        loss_fn (LossFunction): Loss computation function, may be compiled with torch.compile.
        optimizers (OptimizersContainer): Wrapper around one optimizer per model part.
            Handles state dict flattening for PP compatibility.
        lr_schedulers (LRSchedulersContainer): Learning rate schedulers matching optimizers.
        validator (BaseValidator): Optional validation evaluator.
        metrics_processor (MetricsProcessor): Handles logging to console/TensorBoard/W&B.
        model_args (BaseModelArgs): Model architecture hyperparameters (hidden dim,
            num layers, num heads, etc.)

        checkpointer (CheckpointManager): Handles save/load of all training state
            including async checkpointing and HuggingFace format conversion.
        ft_manager (FTManager): TorchFT fault tolerance manager for elastic training.
            Enables recovery from worker failures without losing progress.

        device (torch.device): This rank's assigned GPU device.
        gc_handler (GarbageCollection): Controlled garbage collection to avoid
            stragglers (some ranks doing GC while others wait).
        train_context (TrainContext): Context manager for loss parallel and CP contexts.
        gradient_accumulation_steps (int): Number of micro-batches to accumulate
            before optimizer step. Computed from global_batch_size / (local_batch_size * DP_degree).
        pp_has_first_stage (bool): True if this rank owns the first pipeline stage
            (receives input tokens). Only relevant when PP is enabled.
        pp_has_last_stage (bool): True if this rank owns the last pipeline stage
            (computes loss). Only relevant when PP is enabled.

        step (int): Current training step (0-indexed before first step).
            This is checkpointed and restored on resumption.
        ntokens_seen (int): Cumulative tokens processed across all steps.
            Useful for learning rate schedules based on tokens rather than steps.
    """

    # =========================================================================
    # CORE CONFIGURATION
    # These define what and how we're training
    # =========================================================================
    job_config: JobConfig
    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec

    # =========================================================================
    # SWAPPABLE TRAINING COMPONENTS (from TrainSpec)
    # These vary by model architecture and can be customized
    # =========================================================================
    tokenizer: train_spec_module.BaseTokenizer | None
    dataloader: train_spec_module.BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer
    validator: train_spec_module.BaseValidator
    metrics_processor: train_spec_module.MetricsProcessor
    model_args: train_spec_module.BaseModelArgs

    # =========================================================================
    # NON-SWAPPABLE TRAINING COMPONENTS
    # Core infrastructure that doesn't change by model type
    # =========================================================================
    checkpointer: CheckpointManager
    ft_manager: FTManager

    # =========================================================================
    # RUNTIME UTILITIES
    # Helpers for the training loop
    # =========================================================================
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # =========================================================================
    # TRAINING STATE (checkpointed)
    # These are saved/restored with checkpoints for resumption
    # =========================================================================
    step: int
    ntokens_seen: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    # The @record decorator captures stack traces on failure for better debugging
    # in distributed settings where errors may be hard to trace back.
    @record
    def __init__(self, job_config: JobConfig):
        """
        Initialize the Trainer with the given job configuration.

        This constructor performs the complete setup sequence:
        1. Device initialization and distributed setup
        2. Model construction on meta device (no memory allocated yet)
        3. Model conversion (e.g., FP8 linear layers)
        4. Parallelization (TP, AC, compile, FSDP/PP)
        5. Weight initialization and materialization
        6. Optimizer and LR scheduler creation
        7. Checkpoint loading if available

        Args:
            job_config (JobConfig): Complete training configuration.

        Raises:
            RuntimeError: If PP is enabled but model doesn't support pipelining.
            AssertionError: If batch size configuration is invalid.
        """
        # Log API usage for telemetry (helps PyTorch team understand feature adoption)
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        # =====================================================================
        # STEP 1: CUSTOM IMPORTS
        # Allows users to register custom TrainSpecs or extend functionality
        # before the training setup begins. The module is imported for its
        # side effects (e.g., calling register_train_spec).
        # =====================================================================
        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        # =====================================================================
        # STEP 2: DEVICE INITIALIZATION
        # Each process (launched by torchrun) gets a LOCAL_RANK environment
        # variable indicating which GPU on this node it should use.
        # This ensures each process uses a different GPU.
        # =====================================================================
        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # LOCAL_RANK is set by torchrun; we pin each process to its assigned device.
        # IMPORTANT: Device must be set BEFORE creating TorchFT manager, as FT needs
        # to know which device this rank will use for its communication groups.
        # pyrefly: ignore [missing-attribute]
        device_module.set_device(self.device)

        # =====================================================================
        # STEP 3: DISTRIBUTED INITIALIZATION
        # Initialize NCCL process groups and build the device mesh.
        # The device mesh is a multi-dimensional grid of GPUs that maps
        # parallelism dimensions to physical devices.
        # =====================================================================
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Determine data parallel degree and this rank's position within DP.
        # The "batch" mesh includes both dp_replicate and dp_shard dimensions,
        # representing all ranks that process different portions of the data.
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            # No data parallelism - single device or only other parallelisms
            batch_degree, batch_rank = 1, 0

        # =====================================================================
        # STEP 4: FAULT TOLERANCE SETUP (TorchFT)
        # TorchFT enables elastic training where workers can fail and rejoin.
        # It can dynamically adjust DP groups, so we let it potentially
        # override batch_degree and batch_rank.
        # =====================================================================
        # pyrefly: ignore [bad-argument-type]
        self.ft_manager = FTManager(job_config.fault_tolerance)
        batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)
        # TorchFT can shrink/expand the effective DP groups at runtime when
        # workers fail or rejoin. This affects data loading distribution.

        # =====================================================================
        # STEP 5: GARBAGE COLLECTION CONTROL
        # In distributed training, if one rank does GC while others don't,
        # it causes a "straggler" that slows down collectives. By controlling
        # GC timing, we ensure all ranks GC at the same steps.
        # =====================================================================
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq,  # GC every N steps
            debug=job_config.training.gc_debug,   # Enable GC debugging to find leaks
        )

        # =====================================================================
        # STEP 6: DETERMINISM AND RANDOM SEEDS
        # For reproducibility, we set seeds consistently across ranks.
        # PP ranks get different seeds (distinct_seed_mesh_dims=["pp"]) because
        # they may need different dropout patterns for their different layers.
        # =====================================================================
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            job_config.debug,
            distinct_seed_mesh_dims=["pp"],  # Each PP stage gets its own seed
        )

        # =====================================================================
        # STEP 7: GET MODEL SPECIFICATION
        # TrainSpec bundles all model-specific components. This allows the
        # same Trainer to work with different architectures (Llama, DeepSeek, etc.)
        # =====================================================================
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # =====================================================================
        # STEP 8: BUILD TOKENIZER AND DATALOADER
        # Tokenizer converts text to token IDs. Dataloader handles data
        # loading/batching with proper sharding for distributed training.
        # =====================================================================
        # Some models (e.g., image generation like Flux) don't need a tokenizer
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        # The dataloader is sharded by dp_rank/dp_world_size so each DP rank
        # sees a different subset of the data. This is the key to data parallelism:
        # each rank processes different data, computes gradients, then averages them.
        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=batch_degree,  # Total number of DP ranks
            dp_rank=batch_rank,          # This rank's position in the DP dimension
            tokenizer=self.tokenizer,
            job_config=job_config,
        )

        # =====================================================================
        # STEP 9: LOAD MODEL CONFIGURATION
        # model_args contains architecture hyperparameters (hidden_dim, n_layers, etc.)
        # We look up the "flavor" (e.g., "llama3_8b", "llama3_70b") from the config
        # to get the specific architecture settings.
        # =====================================================================
        model_args = self.train_spec.model_args[job_config.model.flavor]
        # Update model args with job-specific settings (e.g., sequence length
        # may be overridden from the default, or CP may require attention type changes)
        model_args.update_from_config(job_config)
        self.model_args = model_args

        logger.info(
            f"Building {job_config.model.name} {job_config.model.flavor}"
            f"with {json.dumps(dataclasses.asdict(model_args), indent=2, ensure_ascii=False)}"
        )

        # =====================================================================
        # STEP 10: CONSTRUCT MODEL ON META DEVICE
        # Meta device is a "virtual" device where tensors have shapes but no data.
        # This is critical for large models that won't fit in single-GPU memory.
        #
        # WHY META DEVICE?
        # - Allows defining model architecture without allocating memory
        # - After parallelization (TP, FSDP), each rank only materializes its shard
        # - Avoids the need for enormous host memory during initialization
        #
        # The model will be materialized later via `to_empty(device=...)` followed
        # by `init_weights()` which initializes only the local shard.
        # =====================================================================
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            # Meta device avoids immediate memory allocation; weights are materialized later.
            # set_default_dtype ensures new tensors use the configured precision
            # (e.g., bfloat16 for mixed precision training)
            model = self.train_spec.model_cls(model_args)

        # =====================================================================
        # STEP 11: APPLY MODEL CONVERTERS
        # Model converters transform the model before parallelization.
        # Common use cases:
        # - FP8 training: Replace nn.Linear with Float8Linear for 8-bit compute
        # - Fused layers: Replace standard attention with flash attention variants
        # - Quantization: Apply QAT (Quantization-Aware Training) modifications
        #
        # IMPORTANT: Converters must run BEFORE parallelization because:
        # 1. They may change parameter shapes (e.g., FP8 adds scale tensors)
        # 2. Parallelization (TP, FSDP) relies on the final module structure
        # =====================================================================
        model_converters = build_model_converters(job_config, parallel_dims)
        # pyrefly: ignore [bad-argument-type]
        model_converters.convert(model)  # In-place modification of the model
        # Converters may replace modules (e.g., FP8 Linear) before parallelism is applied.

        # =====================================================================
        # STEP 12: SET UP METRICS LOGGING
        # Metrics processor handles logging to console, TensorBoard, and W&B.
        # It tracks loss, throughput (tokens/sec, MFU), memory usage, etc.
        # =====================================================================
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, model_args
        )
        color = self.metrics_processor.color  # ANSI color codes for pretty logging

        # Calculate model size and theoretical FLOPS for MFU (Model FLOPS Utilization)
        # MFU = actual_flops / theoretical_peak_flops, a key efficiency metric
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
            # pyrefly: ignore [bad-argument-type]
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {job_config.model.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # =====================================================================
        # STEP 13: DETERMINE INITIALIZATION DEVICE
        # Choose where to materialize model weights based on training mode:
        # - Seed checkpoint: CPU (single device, no parallelism)
        # - CPU offload: CPU (parameters live on CPU, moved to GPU when needed)
        # - Normal training: GPU (parameters stay on GPU)
        #
        # buffer_device handles non-parameter buffers (e.g., RoPE frequencies)
        # which should stay on GPU even when CPU offloading parameters.
        # =====================================================================
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type  # Keep buffers on GPU for fast access
        else:
            init_device = device_type
            buffer_device = None
        # `init_device` controls where parameters are materialized; `buffer_device` keeps buffers on GPU.

        # =====================================================================
        # STEP 14: BUILD LOSS FUNCTION
        # The loss function may be specialized for distributed training:
        # - Loss parallel: Shards vocabulary dimension across TP ranks
        # - TorchFT integration: Loss computation compatible with fault tolerance
        # =====================================================================
        self.loss_fn = self.train_spec.build_loss_fn(
            job_config, parallel_dims=parallel_dims, ft_manager=self.ft_manager
        )

        # =====================================================================
        # STEP 15: VALIDATE BATCH SIZES AND COMPUTE GRADIENT ACCUMULATION
        #
        # BATCH SIZE TERMINOLOGY:
        # - local_batch_size: Samples processed per forward pass per GPU
        # - global_batch_size: Total samples across all GPUs per optimizer step
        # - gradient_accumulation_steps: Forward passes before one backward+update
        #
        # RELATIONSHIP:
        # global_batch_size = local_batch_size * dp_degree * gradient_accumulation_steps
        #
        # WHY GRADIENT ACCUMULATION?
        # When global_batch_size is too large to fit in GPU memory as a single
        # forward pass, we split it into smaller micro-batches. Gradients from
        # each micro-batch accumulate, then one optimizer step updates weights.
        # This simulates a larger batch size without the memory cost.
        #
        # EXAMPLE:
        # - global_batch_size=1024, local_batch_size=8, dp_degree=32
        # - gradient_accumulation = 1024 / (8 * 32) = 4
        # - Each GPU does 4 forward/backward passes, then one optimizer step
        # =====================================================================
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # Negative value means "auto": use the minimum global batch size
            # that results in exactly 1 gradient accumulation step.
            global_batch_size = job_config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (job_config.training.local_batch_size * batch_degree)
            == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # Calculate how many micro-batches to accumulate before optimizer step
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # IMPORTANT: Scale the loss by 1/gradient_accumulation_steps
        # This is because gradients from multiple micro-batches are summed (not averaged).
        # Without scaling, effective learning rate would increase with more accumulation steps.
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )
        # Loss is scaled down to keep gradient magnitudes consistent across accumulation steps.

        # =====================================================================
        # STEP 16: APPLY PARALLELIZATION AND WEIGHT INITIALIZATION
        #
        # This is the core of distributed training setup. We apply parallelism
        # strategies in a specific order:
        #
        # ORDER OF OPERATIONS (for non-PP):
        # 1. Tensor Parallelism (TP) - Shards weight matrices across TP ranks
        # 2. Activation Checkpointing (AC) - Wraps layers to recompute activations
        # 3. torch.compile - Compiles each transformer block for performance
        # 4. FSDP/DDP - Wraps for data parallel gradient synchronization
        #
        # ORDER OF OPERATIONS (for PP):
        # 1. Split model into pipeline stages
        # 2. Apply TP, AC, compile, FSDP to each stage
        #
        # After parallelization, the model is on meta device with DTensor wrappers.
        # We then materialize weights via to_empty() + init_weights().
        # =====================================================================
        if parallel_dims.pp_enabled:
            # Pipeline Parallelism enabled - split model across PP stages
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {job_config.model.name} "
                    f"does not support pipelining"
                )

            # Split the model into stages and apply all parallelisms
            # Returns:
            # - pp_schedule: Orchestrates micro-batch flow through pipeline
            # - model_parts: List of model chunks owned by this rank
            # - pp_has_first_stage: True if this rank receives input tokens
            # - pp_has_last_stage: True if this rank computes final loss
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # The original `model` is split; we now work with model_parts
            del model

            # Materialize weights for each model part (stage)
            for m in self.model_parts:
                # to_empty() allocates storage without initializing values
                m.to_empty(device=init_device)
                with torch.no_grad():
                    # init_weights() initializes only the local shard's values
                    # For DTensor-wrapped params, this respects the sharding
                    # pyrefly: ignore [not-callable]
                    m.init_weights(buffer_device=buffer_device)
                m.train()  # Set to training mode

            # Warn if logging rank won't see loss (only last PP stage has loss)
            # pyrefly: ignore [bad-argument-type]
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # No Pipeline Parallelism - apply other parallelisms directly
            # This applies: TP -> AC -> compile -> FSDP/DDP
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)

            # Materialize weights on the target device
            model.to_empty(device=init_device)
            with torch.no_grad():
                # pyrefly: ignore [not-callable]
                model.init_weights(buffer_device=buffer_device)
            model.train()

            # Wrap in list for consistent interface with PP case
            self.model_parts = [model]

        # Set up fault tolerance hooks for gradient synchronization
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # =====================================================================
        # STEP 17: MEMORY MONITORING AND MFU CALCULATION SETUP
        # Track GPU memory usage and calculate theoretical peak FLOPS for MFU.
        # MFU (Model FLOPS Utilization) = actual_flops / peak_flops
        # A good MFU is typically 30-50% depending on model and hardware.
        # =====================================================================
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # =====================================================================
        # STEP 18: BUILD OPTIMIZER AND LR SCHEDULER
        #
        # IMPORTANT: Optimizer must be built AFTER parallelization because:
        # 1. FSDP wraps parameters; optimizer needs the wrapped params
        # 2. Parameter groups may differ between model parts (for PP)
        # 3. Optimizer state (momentum, etc.) needs to match sharded params
        #
        # OptimizersContainer wraps multiple optimizers (one per model_part)
        # to provide a unified interface for PP where each stage has its
        # own optimizer.
        # =====================================================================
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )

        # Register post-optimizer-step hook for model converters.
        # This is critical for FP8 training where we need to:
        # - Compute dynamic scaling factors (amax) for all parameters
        # - Issue a single all-reduce for efficiency (rather than per-layer)
        # The hook runs after optimizer.step() but before the next forward pass.
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        # Give metrics processor access to optimizers for learning rate logging
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # =====================================================================
        # STEP 19: INITIALIZE TRAINING STATE
        # These values are checkpointed and restored on resumption.
        # Must be initialized BEFORE checkpoint loading so load_state_dict works.
        # =====================================================================
        self.step = 0              # Current training step (0 = before first step)
        self.ntokens_seen = 0      # Cumulative tokens processed

        # =====================================================================
        # STEP 20: SET UP CHECKPOINT MANAGER
        # CheckpointManager handles:
        # - Distributed checkpointing with DCP (Distributed Checkpoint)
        # - Async checkpointing to avoid blocking training
        # - HuggingFace format conversion for model export
        # - Loading from checkpoints or pre-trained weights
        # - Fault tolerance checkpoint coordination
        #
        # All stateful components are passed to the checkpointer:
        # - model_parts: Model weights (with FSDP/TP sharding metadata)
        # - optimizers: Optimizer states (momentum, variance for Adam)
        # - lr_schedulers: LR scheduler states (current epoch, etc.)
        # - dataloader: Data iterator position (for exact resumption)
        # - train_state (self): step counter and ntokens_seen
        # =====================================================================
        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},  # Trainer itself is stateful
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                # State dict adapter converts between native and HF formats
                self.train_spec.state_dict_adapter(
                    model_args, job_config.model.hf_assets_path
                )
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
        )

        # =====================================================================
        # STEP 21: SET UP TRAINING CONTEXT MANAGERS
        #
        # train_context: Enables loss parallel and context parallel scopes.
        # - Loss parallel: Shards cross-entropy across TP dimension
        #   (vocabulary is large, so sharding saves memory/compute)
        #
        # maybe_enable_amp: Automatic Mixed Precision context.
        # - Casts forward pass to lower precision (FP16/BF16)
        # - Gradients computed in lower precision
        # - Master weights kept in FP32 for stability
        # =====================================================================
        loss_parallel_enabled = (
            parallel_dims.tp_enabled
            and not job_config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if job_config.validation.enable:
            assert self.train_spec.build_validator_fn is not None

            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        """
        Initialize distributed training environment and create device mesh.

        This method:
        1. Initializes NCCL process groups for GPU communication
        2. Sets up optional CPU backend for CPU offloading
        3. Creates ParallelDims which manages the multi-dimensional device mesh

        DEVICE MESH EXPLAINED:
        ----------------------
        The device mesh is a multi-dimensional logical grid that maps physical GPUs
        to parallelism dimensions. For example, with 16 GPUs and TP=2, PP=2, DP=4:

        Physical GPUs:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        Logical mesh (3D):
            - TP dimension: Groups of 2 GPUs share tensor-sharded weights
            - PP dimension: Groups of 2 GPUs form a pipeline (different layers)
            - DP dimension: Groups of 4 GPUs see different data

        This allows combining multiple parallelism strategies efficiently.

        Returns:
            ParallelDims: Object managing device mesh and providing sub-meshes
                for each parallelism dimension.
        """
        job_config = self.job_config
        world_size = dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )

        parallelism_config = job_config.parallelism
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,      # FSDP sharding
            dp_replicate=parallelism_config.data_parallel_replicate_degree,  # DDP replication
            cp=parallelism_config.context_parallel_degree,               # Context/sequence parallel
            tp=parallelism_config.tensor_parallel_degree,                # Tensor parallel
            pp=parallelism_config.pipeline_parallel_degree,              # Pipeline parallel
            ep=parallelism_config.expert_parallel_degree,                # Expert parallel (MoE)
            etp=parallelism_config.expert_tensor_parallel_degree,        # Expert tensor parallel
            world_size=world_size,
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """
        Generator that fetches and preprocesses batches from the dataloader.

        This generator:
        1. Fetches batches from the underlying dataloader
        2. Moves tensors to the appropriate device (GPU)
        3. Tracks data loading time for performance metrics
        4. Counts tokens for throughput calculation

        The generator handles StopIteration gracefully by raising
        DataloaderExhaustedError, which allows the training loop to
        exit cleanly when data runs out.

        Args:
            data_iterable: Iterable yielding (input_dict, labels) tuples

        Yields:
            tuple: (input_dict, labels) with tensors on the correct device

        Raises:
            DataloaderExhaustedError: When the dataloader is exhausted
        """
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed. We raise a custom
                # exception to allow proper handling in the training loop.
                raise DataloaderExhaustedError() from ex

            input_dict, labels = batch
            # Count tokens for throughput metrics (tokens/sec calculation)
            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Move tensors from CPU to GPU
            # This is typically fast due to pinned memory in DataLoader
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        attn_type = getattr(self.model_args, "attn_type", "sdpa")
        if attn_type in ["flex", "varlen"]:
            # pyrefly: ignore [not-callable]
            extra_kwargs["attention_masks"] = self.model_parts[0].get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs=extra_inputs,
            )

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute one forward and backward pass for a single micro-batch.

        This is the core computational step that:
        1. Processes input through post_dataloading_process
        2. Sets up context parallelism if enabled
        3. Runs forward pass (with optional PP schedule)
        4. Computes loss
        5. Runs backward pass to compute gradients

        CONTEXT PARALLELISM (CP):
        -------------------------
        CP splits the sequence dimension across GPUs. This requires:
        - Splitting input tensors along sequence dimension
        - Handling RoPE (freqs_cis) buffer appropriately
        - Communication during attention (ring attention pattern)

        PIPELINE PARALLELISM (PP):
        --------------------------
        With PP enabled, pp_schedule.step() handles both forward and backward:
        - Micro-batches flow through pipeline stages
        - Each stage runs forward, sends activations to next stage
        - Backward pass flows in reverse
        - Schedule (1F1B, Interleaved) determines ordering

        NON-PP PATH:
        ------------
        Without PP, this is a simple forward -> loss -> backward sequence.
        AMP context wraps the forward pass for mixed precision.

        Args:
            input_dict: Dictionary containing input tensors from dataloader
            labels: Target labels for loss computation

        Returns:
            torch.Tensor: Computed loss value (scalar or per-GPU for PP)
        """
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        # =====================================================================
        # CONTEXT PARALLELISM SETUP
        # CP requires splitting sequence-dimension tensors across CP ranks.
        # We identify all buffers that need splitting (inputs, labels, freqs_cis)
        # and their sequence dimensions for proper sharding.
        # =====================================================================
        cp_buffers: list[torch.Tensor] = [inputs, labels]
        cp_seq_dims = [1, 1]  # Sequence dim is typically dim 1 for [batch, seq, ...]
        if hasattr(model_parts[0], "freqs_cis"):
            # RoPE frequencies need special handling - they're indexed by position
            for m in model_parts:
                assert isinstance(m.freqs_cis, torch.Tensor)
                cp_buffers.append(m.freqs_cis)
            cp_seq_dims += [0 for _ in model_parts]  # freqs_cis has seq on dim 0
        # CP treats inputs/labels as seq-dim=1 and buffers like freqs_cis as seq-dim=0.

        optional_context_parallel_ctx = None
        if parallel_dims.cp_enabled:
            cp_mesh = parallel_dims.get_mesh("cp")
            optional_context_parallel_ctx = dist_utils.create_context_parallel_ctx(
                cp_mesh=cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=cp_seq_dims,
                cp_no_restore_buffers={inputs, labels},  # Don't restore after forward
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )

        if parallel_dims.pp_enabled:
            # =====================================================================
            # PIPELINE PARALLEL PATH
            # pp_schedule.step() orchestrates the entire forward/backward flow:
            # - Splits batch into micro-batches
            # - Coordinates send/recv between pipeline stages
            # - Handles gradient accumulation across micro-batches
            # Only the LAST stage computes loss; others just forward activations.
            # =====================================================================
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                # Only the last stage computes loss; other stages pass None.
                if self.pp_has_first_stage:
                    # First stage receives input tokens
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    # Middle/last stages receive activations from previous stage
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            # Accumulate losses across pipeline micro-batches
            # Note: We use sum (not mean) because loss is already scaled down
            # by n_microbatches in pipeline_parallel.py
            loss = (
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)  # Sentinel for non-loss ranks
            )
        else:
            # =====================================================================
            # NON-PP PATH (standard forward/backward)
            # This is the simple case: forward -> loss -> backward
            # AMP context handles mixed precision casting
            # =====================================================================
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    loss = self.loss_fn(pred, labels)
                # IMPORTANT: Free prediction tensor before backward pass.
                # This reduces peak memory by freeing the large logits tensor
                # (vocab_size * batch * seq) before backward allocates gradients.
                del pred
                loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        """
        Execute one complete training step with gradient accumulation.

        A training step consists of:
        1. Zero gradients from previous step
        2. Accumulate gradients over multiple micro-batches (gradient accumulation)
        3. Clip gradients (with PP-aware reduction if needed)
        4. Update weights (optimizer.step)
        5. Update learning rate (lr_scheduler.step)
        6. Log metrics (loss, throughput, grad norm, etc.)

        GRADIENT ACCUMULATION:
        ----------------------
        When global_batch_size > local_batch_size * dp_degree, we need to
        accumulate gradients over multiple forward/backward passes before
        updating weights. This simulates training with a larger batch size
        than can fit in memory.

        The loss is pre-scaled (in __init__) to account for accumulation,
        so gradients from accumulated micro-batches can simply be summed.

        GRADIENT CLIPPING:
        ------------------
        Gradient norms are clipped to prevent exploding gradients.
        With PP, gradient norms must be reduced across pipeline stages
        before clipping, as different stages see different layers' gradients.

        Args:
            data_iterator: Iterator yielding (input_dict, labels) tuples
        """
        # Clear gradients from previous step
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        parallel_dims = self.parallel_dims

        # =====================================================================
        # GRADIENT ACCUMULATION LOOP
        # Process multiple micro-batches, accumulating gradients.
        # The loss is already scaled by 1/gradient_accumulation_steps,
        # so the final gradient magnitude is correct.
        # =====================================================================
        accumulated_losses = []
        for _microbatch in range(self.gradient_accumulation_steps):
            # pyrefly: ignore [no-matching-overload]
            input_dict, labels = next(data_iterator)
            loss = self.forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss.detach())

        # =====================================================================
        # GRADIENT CLIPPING
        # Clip gradient norms to prevent exploding gradients.
        # For PP, we need to reduce gradient norms across stages first,
        # as each stage only sees gradients for its own layers.
        # =====================================================================
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,  # Use faster foreach implementation
            pp_mesh=parallel_dims.get_optional_mesh("pp"),  # For cross-stage reduction
            ep_enabled=parallel_dims.ep_enabled,  # Expert parallel needs special handling
        )

        # Wait for async checkpoint staging if in progress
        # This ensures checkpoint data is fully staged before we modify weights
        self.checkpointer.maybe_wait_for_staging()

        # =====================================================================
        # OPTIMIZER AND LR SCHEDULER STEP
        # Update model weights and advance learning rate schedule.
        # =====================================================================
        self.optimizers.step()
        self.lr_schedulers.step()

        # Sum losses across gradient accumulation steps for logging
        # (already scaled, so sum is appropriate)
        loss = torch.sum(torch.stack(accumulated_losses))

        # =====================================================================
        # METRICS LOGGING
        # Only log on configured intervals to avoid overhead.
        # With DP/CP, we reduce loss across all data-parallel ranks
        # to get global average and max loss.
        # =====================================================================
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            # Reduce loss across data parallel ranks for global statistics
            loss = loss.detach()
            ft_pg = self.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_mean(loss, loss_mesh, ft_pg),
                dist_utils.dist_max(loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
        else:
            # Single rank or non-DP case
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        """
        Main training loop.

        This method:
        1. Loads checkpoint if available (resumes training state)
        2. Sets up profiling and memory snapshots
        3. Runs the training loop until completion
        4. Handles checkpointing, validation, and profiling

        TRAINING LOOP FLOW:
        -------------------
        for each step:
            1. Run garbage collection (controlled timing)
            2. Execute train_step (forward, backward, optimizer step)
            3. Save checkpoint if interval reached
            4. Run validation if interval reached
            5. Step profilers

        PROFILING:
        ----------
        - torch_profiler: PyTorch profiler for performance analysis
        - memory_profiler: CUDA memory snapshots for debugging OOM

        TIMEOUT ADJUSTMENT:
        -------------------
        After the first step completes (including lazy init and compilation),
        we reduce the NCCL timeout. The initial longer timeout allows for
        slow first-step compilation; subsequent steps should be faster.

        Raises:
            DataloaderExhaustedError: When training data runs out (handled gracefully)
        """
        job_config = self.job_config

        # Load checkpoint - this updates self.step, optimizer state, etc.
        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")
        # CheckpointManager updates self.step and any stateful components.

        # For fault tolerance, each replica saves to its own subfolder
        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )

        # =====================================================================
        # TRAINING CONTEXT MANAGERS
        # Set up profiling and optional semi-sync training for fault tolerance
        # =====================================================================
        with (
            # PyTorch profiler for performance analysis (trace, memory, etc.)
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as torch_profiler,
            # CUDA memory snapshot for debugging OOM issues
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as memory_profiler,
            # Semi-synchronous training for fault tolerance
            # Allows some ranks to be ahead while others catch up
            maybe_semi_sync_training(
                # pyrefly: ignore [bad-argument-type]
                job_config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    self.model_args.n_layers
                    if hasattr(self.model_args, "n_layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    self.train_spec.fragment_fn
                    if hasattr(self.train_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            # =====================================================================
            # MAIN TRAINING LOOP
            # =====================================================================
            # pyrefly: ignore [bad-argument-type]
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1

                # Run controlled GC to avoid straggler effects
                self.gc_handler.run(self.step)

                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    # Data ran out - exit gracefully
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # Save checkpoint at configured intervals and on last step
                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # Run validation if configured and interval reached
                if (
                    self.job_config.validation.enable
                    and self.validator.should_validate(self.step)
                ):
                    # Disable loss rescaling during validation
                    # (validation uses full batch, not gradient accumulation)
                    # pyrefly: ignore [missing-attribute]
                    with self.loss_fn.no_rescale():
                        # pyrefly: ignore [bad-argument-count]
                        self.validator.validate(self.model_parts, self.step)

                # Advance profilers to next step
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # TIMEOUT OPTIMIZATION:
                # After the first step (which includes lazy initialization,
                # torch.compile, and NCCL warmup), reduce the timeout.
                # This allows faster failure detection for subsequent steps.
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        parallel_dims=self.parallel_dims,
                    )

        # Give other ranks time to finish before destroying process groups
        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.job_config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()


def main(trainer_class: type[Trainer]) -> None:
    """Main entry point for training with a specified trainer class.

    Args:
        trainer_class: The trainer class to instantiate (e.g., Trainer, FluxTrainer, TorchCommsTrainer)
    """
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    try:
        trainer = trainer_class(config)

        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_weights() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main(Trainer)
