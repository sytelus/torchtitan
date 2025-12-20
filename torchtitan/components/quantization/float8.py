# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 (8-bit Floating Point) Training Support
=============================================

This module implements FP8 training support for TorchTitan using the torchao library.
FP8 training uses 8-bit floating point formats for matrix multiplications during
forward and backward passes, reducing memory bandwidth and enabling tensor core
acceleration on supported hardware.

FP8 FORMATS OVERVIEW:
---------------------
FP8 uses two main formats optimized for different use cases:
- E4M3 (4 exponent bits, 3 mantissa): Higher precision, smaller range. Used for activations/weights.
- E5M2 (5 exponent bits, 2 mantissa): Lower precision, larger range. Used for gradients.

DYNAMIC SCALING:
----------------
FP8 has a very limited dynamic range (~240 for E4M3 vs ~65504 for FP16).
To prevent overflow/underflow, values are scaled before FP8 conversion:

    fp8_value = original_value * scale_factor

The scale factor is computed dynamically based on the absolute maximum (amax)
of the tensor, ensuring values fit within FP8 range.

TWO SCALING APPROACHES:
-----------------------
1. **Delayed Scaling**: Scale is computed from previous iteration's amax (cached).
   - Lower overhead (no extra reduction in forward pass)
   - May cause overflow if values change rapidly between iterations

2. **Dynamic Scaling (Default)**: Scale computed from current tensor's amax.
   - More accurate but requires extra reduce operations
   - Recommended for most training scenarios

FSDP FLOAT8 ALL-GATHER OPTIMIZATION:
------------------------------------
When FSDP is enabled, weights are normally all-gathered in FP16/BF16.
With `enable_fsdp_float8_all_gather`, weights are:
1. Stored in FP8 format (8 bytes per element instead of 16)
2. All-gathered in FP8 (halves communication volume)
3. Converted to higher precision just before compute

This significantly reduces communication time for FSDP workloads.

RECIPES:
--------
Recipes are pre-configured FP8 settings optimized for different scenarios:
- "tensorwise": Standard FP8 with per-tensor scaling (default)
- "rowwise": Per-row scaling for better accuracy (requires specific hardware)

HARDWARE REQUIREMENTS:
----------------------
- SM89+ (H100, H800) for native FP8 compute
- Older hardware can use `emulate=True` for testing (very slow)

USAGE:
------
Enable in config:
```toml
[quantize.linear.float8]
enable = true
enable_fsdp_float8_all_gather = true  # For FSDP communication savings
```

GOTCHAS:
--------
1. FP8 is experimental; expect accuracy differences from FP16/BF16 training
2. Not all layers benefit from FP8 (small layers may be slower due to overhead)
3. Use `filter_fqns` to exclude specific layers from FP8 conversion
4. Recipe "rowwise" requires inductor config for precision cast emulation
"""

from functools import partial

import torch
import torch._inductor.config
import torch.nn as nn
from torchtitan.components.quantization import (
    FP8_GROUP_ALIGNMENT_SIZE,
    QuantizationConverter,
)

from torchtitan.config.job_config import Float8Linear, JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.moe.utils import set_token_group_alignment_size_m
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import module_filter_fn

# Flag for automatic filtering of small layers that don't benefit from FP8
AUTO_FILTER_SMALL_KN_FLAG = "auto_filter_small_kn"


class Float8LinearConverter(QuantizationConverter):
    """
    Model converter that replaces nn.Linear layers with Float8Linear.

    This converter enables FP8 training by swapping standard Linear layers
    with FP8-compatible versions from torchao. The FP8 Linear layers perform
    matrix multiplications in 8-bit floating point format, reducing memory
    bandwidth and enabling tensor core acceleration.

    The converter supports two modes:
    1. Recipe-based: Use pre-configured recipes (e.g., "tensorwise", "rowwise")
    2. Manual configuration: Set individual FP8 options

    Attributes:
        enabled (bool): Whether FP8 conversion is enabled
        config: TorchAO Float8LinearConfig with FP8 settings
        precompute_scale (bool): Whether to precompute scales for FSDP
        filter_fqns (list): Module FQNs to exclude from conversion
        filter_fn (callable): Function to filter which modules to convert

    Example:
        ```python
        converter = Float8LinearConverter(job_config, parallel_dims)
        converter.convert(model)  # Replaces Linear -> Float8Linear
        ```
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        """
        Initialize the Float8 converter.

        Args:
            job_config: Training configuration with FP8 settings
            parallel_dims: Parallelism configuration (needed for FSDP awareness)

        Raises:
            ValueError: If hardware doesn't support FP8 and emulation is disabled
            ImportError: If torchao is not installed
        """
        super().__init__(job_config, parallel_dims)
        float8_config: Float8Linear = job_config.quantize.linear.float8
        compile_config = job_config.compile
        model_compile_enabled = (
            compile_config.enable and "model" in compile_config.components
        )

        # =====================================================================
        # HARDWARE CHECK
        # FP8 compute requires SM89+ (H100, H800). For older hardware,
        # emulation mode can be used for testing (very slow, not for production).
        # Note: Emulation only works in eager mode, not with torch.compile.
        # =====================================================================
        if has_cuda_capability(8, 9) or (
            float8_config.emulate and not model_compile_enabled
        ):
            pass
        else:
            raise ValueError(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later."
                "To enable testing on older hardware, set `float8.emulate` to True in eager mode.",
            )
        try:
            from torchao.float8 import Float8LinearConfig as TorchAOFloat8LinearConfig
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        if float8_config.recipe_name is not None and not hasattr(
            TorchAOFloat8LinearConfig, "from_recipe_name"
        ):
            logger.warning(
                "Failed to swap to Float8Linear with recipe lookup because the torchao version "
                "is too old, please install torchao v0.9.0 or later and try again",
            )
            return

        self.filter_fqns = float8_config.filter_fqns
        self.filter_fn = self._init_filter_fn(float8_config)

        if float8_config.recipe_name is not None:
            # Recipes encapsulate scaling/format choices; they override explicit flags.
            assert not float8_config.enable_fsdp_float8_all_gather, (
                "using `float8_config.enable_fsdp_float8_all_gather` together "
                "with `float8_config.recipe_name` is not supported"
            )

            self.config = TorchAOFloat8LinearConfig.from_recipe_name(
                float8_config.recipe_name
            )
            self.precompute_scale = False
            logger.info(
                f"Float8 training active with recipe {float8_config.recipe_name}"
            )

            # short-term solution for https://github.com/pytorch/pytorch/issues/150859
            if float8_config.recipe_name == "rowwise":
                torch._inductor.config.emulate_precision_casts = True
                logger.debug(
                    "Set torch._inductor.config.emulate_precision_casts to True"
                )
        else:
            # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
            enable_fsdp_float8_all_gather = (
                parallel_dims.dp_shard_enabled
                and float8_config.enable_fsdp_float8_all_gather
            )
            self.config = TorchAOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                emulate=float8_config.emulate,
            )
            # for precompute_float8_dynamic_scale_for_fsdp
            self.precompute_scale = (
                enable_fsdp_float8_all_gather
                and float8_config.precompute_float8_dynamic_scale_for_fsdp
            )
            logger.info("Float8 tensorwise scaled training active")

        self.enabled = True

    def _init_filter_fn(self, float8_config: Float8Linear):
        # use auto_filter if filter_fqns "auto_filter_small_kn" is one of the given fqns.
        use_auto_filter = AUTO_FILTER_SMALL_KN_FLAG in float8_config.filter_fqns
        if use_auto_filter:
            try:
                from torchao.float8 import _auto_filter_for_recipe

                logger.info(
                    "Using _auto_filter_for_recipe to avoid converting linear layers with dims too small "
                    "to benefit from float8 training. See docs/float8.md for more info."
                )

                recipe_name = (
                    float8_config.recipe_name
                    if float8_config.recipe_name
                    else "tensorwise"
                )

                # remove auto filter flag from filter_fqns before passing to _auto_filter_for_recipe
                float8_config.filter_fqns.remove(AUTO_FILTER_SMALL_KN_FLAG)

                return _auto_filter_for_recipe(
                    recipe_name,
                    filter_fqns=float8_config.filter_fqns,
                )
            except ImportError:
                logger.warning(
                    (
                        "Using default module_filter_fn for float8 model conversion. "
                        "To use _auto_filter_for_recipe, please install torchao nightly build."
                    )
                )

        # use default filter func
        return partial(module_filter_fn, filter_fqns=float8_config.filter_fqns)

    def convert(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=self.filter_fn,
        )
        logger.info(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)


class Float8GroupedMMConverter(QuantizationConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        self.fqns = job_config.quantize.grouped_mm.float8.fqns
        compile_config = job_config.compile
        model_compile_enabled = (
            compile_config.enable and "model" in compile_config.components
        )
        if not has_cuda_capability(8, 9):
            raise ValueError("Float8 MoE training only supported on SM89 or later.")

        if not model_compile_enabled:
            logger.warning(
                "Compile is required for high performance float8 MoE training; enable it with --compile.enable"
            )

        # Validate MoE training prototype limitations.
        assert (
            job_config.parallelism.pipeline_parallel_degree == 1
        ), "Float8 MoE training prototype does not yet support pipeline parallelism"
        assert (
            job_config.parallelism.context_parallel_degree == 1
        ), "Float8 MoE training prototype does not yet support context parallelism"

        # For fp8 grouped GEMM, token group sizes must be multiples of 16
        # (16 byte alignment / 1 byte per elem = 16 elements)
        set_token_group_alignment_size_m(FP8_GROUP_ALIGNMENT_SIZE)
        self.enabled = True

    def convert(self, model: nn.Module):
        """
        Mutates the model inplace replacing instances of nn.Parameter with ScaledGroupedMMTensor,
        to perform dynamic float8 rowwise quantization + scaled grouped GEMMs for the target MoE FQNs.
        """
        from torchao.quantization.quant_api import quantize_

        try:
            from torchao.prototype.moe_training.conversion_utils import (
                MoETrainingConfig,
            )
        except ImportError as e:
            raise ImportError(
                "torchao installation does not have MoE training support. Please install torchao nightly build."
            ) from e

        def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
            for target_fqn in self.fqns:
                if target_fqn in cur_fqn:
                    return True
            return False

        config = MoETrainingConfig()
        quantize_(model, config=config, filter_fn=moe_module_filter_fn)
        logger.info(
            f"Converted MoE layers matching FQNS {self.fqns} "
            "to use dynamic float8 rowwise quantization with scaled grouped GEMMs"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        pass


register_model_converter(Float8LinearConverter, "quantize.linear.float8")
register_model_converter(Float8GroupedMMConverter, "quantize.grouped_mm.float8")
