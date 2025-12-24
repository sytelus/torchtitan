"""
GPT-2 Model Arguments
=====================

This module defines the model hyperparameters for GPT-2-style transformers.
GPT-2 differs from Llama in several ways:
- Uses learned positional embeddings (not RoPE)
- Uses GELU activation (not SwiGLU)
- Uses LayerNorm (not RMSNorm)
- Uses weight tying between embeddings and output
"""

from dataclasses import dataclass

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs


@dataclass
class GPT2ModelArgs(BaseModelArgs):
    """Model hyperparameters for GPT-2-style transformers.

    GPT-2 Architecture:
    - 124M: dim=768, n_layers=12, n_heads=12
    - 355M: dim=1024, n_layers=24, n_heads=16
    - 774M: dim=1280, n_layers=36, n_heads=20
    - 1.5B: dim=1600, n_layers=48, n_heads=25
    """

    dim: int = 768
    """Model hidden size (embedding dimension)."""

    n_layers: int = 12
    """Number of transformer layers."""

    n_heads: int = 12
    """Number of attention heads."""

    vocab_size: int = 50257
    """Vocabulary size (GPT-2 tokenizer vocab size)."""

    max_seq_len: int = 1024
    """Maximum sequence length (context window)."""

    dropout: float = 0.0
    """Dropout probability (0.0 for pretraining, >0 for finetuning)."""

    bias: bool = True
    """Whether to use bias in linear layers and LayerNorm."""

    norm_eps: float = 1e-5
    """Epsilon for LayerNorm."""

    weight_tying: bool = True
    """Whether to tie embedding and output weights (GPT-2 does this)."""

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        """Update model args from job config."""
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            # For GPT-2, we can extend context but learned positional embeddings
            # won't generalize well beyond training length
            pass
        self.max_seq_len = seq_len

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        """Calculate number of parameters and FLOPs per token.

        Returns:
            Tuple of (n_params, flops_per_token)
        """
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Calculate FLOPs per forward pass (approximate)
        # For each transformer layer:
        # - Attention: 4 * seq_len * dim^2 (Q,K,V,O projections) + 2 * seq_len^2 * dim (attention scores)
        # - FFN: 8 * seq_len * dim^2 (two linear layers with 4*dim hidden)
        # Total per layer: 12 * seq_len * dim^2 + 2 * seq_len^2 * dim

        d = self.dim
        L = self.n_layers
        s = seq_len

        # Simplified FLOPs calculation (multiply-adds counted as 2 ops)
        attn_flops = 4 * s * d * d + 2 * s * s * d  # per layer
        ffn_flops = 2 * s * d * (4 * d)  # 2 matmuls with 4*dim hidden
        layer_flops = attn_flops + ffn_flops
        total_flops = L * layer_flops

        # Per-token FLOPs
        flops_per_token = total_flops // s

        return n_params, flops_per_token
