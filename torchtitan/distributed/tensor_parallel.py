# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensor Parallelism (TP) Utilities
=================================

This module provides utilities for Tensor Parallelism, specifically for enabling
asynchronous tensor parallel communication.

TENSOR PARALLELISM OVERVIEW:
----------------------------
Tensor Parallelism splits individual layers (weight matrices) across multiple GPUs.
Unlike Data Parallelism where each GPU has a full model copy, TP distributes the
weights of each layer:

For a Linear layer with weight W (d_in x d_out):
- Column Parallel: Split W along columns -> Each GPU has (d_in x d_out/N)
- Row Parallel: Split W along rows -> Each GPU has (d_in/N x d_out)

COMMUNICATION PATTERN:
----------------------
TP requires communication within each transformer block:

    Column-Parallel Linear:
    ┌─────────────┐    All-Gather    ┌─────────────┐
    │  GPU 0: W0  │ ───────────────> │  Full input │
    │  GPU 1: W1  │                  │  on all GPUs│
    └─────────────┘                  └─────────────┘

    Row-Parallel Linear:
    ┌─────────────┐    Reduce-Scatter    ┌─────────────┐
    │ Partial out │ ──────────────────> │  Output 0/1 │
    │  on each    │                      │  sharded    │
    └─────────────┘                      └─────────────┘

Typically, attention uses column-parallel for Q/K/V projection and row-parallel
for output projection. MLP uses column-parallel for up-projection and row-parallel
for down-projection.

ASYNC TENSOR PARALLEL:
----------------------
Standard TP has synchronization points at each all-reduce/all-gather. Async TP
overlaps computation and communication by:
1. Using symmetric memory for faster, lower-overhead communication
2. Enabling micro-pipelining in torch.compile to overlap ops

This can significantly improve throughput, especially for models with many
small TP communications.

REQUIREMENTS FOR ASYNC TP:
--------------------------
1. torch.compile must be enabled for the model
2. Hardware support for symmetric memory (NVLink, etc.)
3. Proper mesh configuration

SEQUENCE PARALLELISM (SP):
--------------------------
When TP is enabled, Sequence Parallelism is typically also used. SP keeps the
sequence dimension sharded during non-TP operations (LayerNorm, dropout) to
reduce memory. The sequence is gathered only for TP operations.

Note: The actual TP parallelization is done in the model's parallelize_fn
(see llama3/infra/parallelize.py for example). This module only handles
async TP configuration.
"""


import torch
import torch._inductor.config
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def maybe_enable_async_tp(job_config: JobConfig, tp_mesh: DeviceMesh):
    """
    Enable asynchronous tensor parallel communication if configured.

    Async TP overlaps communication with computation by using:
    1. Symmetric memory: Low-overhead GPU-to-GPU communication
    2. Micro-pipelining: torch.compile optimization to overlap ops

    This can provide 5-15% throughput improvement for TP-heavy workloads.

    REQUIREMENTS:
    - job_config.parallelism.enable_async_tensor_parallel = True
    - job_config.compile.enable = True
    - "model" in job_config.compile.components

    Args:
        job_config: Training configuration
        tp_mesh: DeviceMesh for the TP dimension

    Raises:
        RuntimeError: If async TP is enabled but torch.compile is not
            configured for the model component.
    """
    if not job_config.parallelism.enable_async_tensor_parallel:
        return

    # Async TP REQUIRES torch.compile because micro-pipelining is a
    # compiler optimization that reorders operations to overlap comm/compute
    if not (job_config.compile.enable and "model" in job_config.compile.components):
        raise RuntimeError(
            "Async TP requires 'model' in --compile.components and --compile.enable"
        )

    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    # SYMMETRIC MEMORY:
    # A low-overhead communication mechanism that allows GPUs to directly
    # read/write each other's memory without explicit synchronization.
    # This is much faster than traditional NCCL collectives for small messages.
    #
    # MICRO-PIPELINING:
    # torch.compile reorders operations so that communication (all-gather,
    # reduce-scatter) happens in parallel with independent computation.
    # Instead of: compute -> comm -> compute
    # We get:     compute --\
    #                        +-- comm overlapped
    #             compute --/
    torch._inductor.config._micro_pipeline_tp = True
    enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info("Async TP is enabled")
