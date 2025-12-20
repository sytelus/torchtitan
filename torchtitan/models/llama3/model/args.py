# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class RoPEScalingArgs:
    """RoPE scaling parameters for long-context extrapolation."""

    scaling_factor: float = 8.0
    """Overall RoPE scaling factor (higher enables longer contexts)."""

    low_freq_factor: float = 1.0
    """Scaling factor for low-frequency components."""

    high_freq_factor: float = 4.0
    """Scaling factor for high-frequency components."""

    original_max_position_embeddings: int = 8192
    """Base context length the model was trained with."""


@dataclass
class TransformerModelArgs(BaseModelArgs):
    """Model hyperparameters for Llama-style dense transformer blocks."""

    dim: int = 4096
    """Model hidden size (embedding dimension)."""

    n_layers: int = 32
    """Number of transformer layers (blocks)."""

    n_heads: int = 32
    """Number of attention heads."""

    n_kv_heads: int | None = None
    """Number of key/value heads (if using GQA); None means use `n_heads`."""

    vocab_size: int = 128256
    """Vocabulary size for token embeddings and output projection."""

    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    """Round FFN hidden size up to a multiple of this (hardware-friendly)."""

    ffn_dim_multiplier: float | None = None
    """Optional multiplier for FFN hidden size before rounding to `multiple_of`."""

    norm_eps: float = 1e-5
    """Epsilon for RMSNorm/LayerNorm."""

    rope_theta: float = 10000
    """RoPE base theta; controls rotation frequency spectrum."""

    rope_scaling_args: RoPEScalingArgs = field(default_factory=RoPEScalingArgs)
    """RoPE scaling config for long-context support."""

    max_seq_len: int = 131072
    """Maximum sequence length supported by the model config."""

    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    """Use depth-aware init (layer-specific scaling) when true."""

    attn_type: str = "sdpa"
    """Attention implementation: 'sdpa', 'flex', or other model-defined options."""

    attn_mask_type: str = "causal"
    """Attention mask type (typically 'causal' for language models)."""

    eos_id: int = 0
    """End-of-sequence token id used by the tokenizer."""

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if (
            job_config.parallelism.context_parallel_degree > 1
            and self.attn_type != "sdpa"
        ):
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_dense_model_nparams_and_flops(
            self,
            model,
            2 * (self.dim // self.n_heads),
            seq_len,
        )
