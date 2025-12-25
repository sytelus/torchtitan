"""
GPT-2 Model Implementation
==========================

A clean implementation of GPT-2 following the original architecture:
- Learned positional embeddings
- Pre-LayerNorm architecture (norm before attention/FFN)
- GELU activation in FFN
- Optional weight tying between embeddings and output

This implementation prioritizes readability and follows TorchTitan conventions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.train_spec import ModelProtocol

from .args import GPT2ModelArgs


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional dropout."""

    def __init__(self, model_args: GPT2ModelArgs):
        super().__init__()
        assert model_args.dim % model_args.n_heads == 0

        self.n_heads = model_args.n_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.dim = model_args.dim

        # Key, Query, Value projections combined
        self.c_attn = nn.Linear(model_args.dim, 3 * model_args.dim, bias=model_args.bias)
        # Output projection
        self.c_proj = nn.Linear(model_args.dim, model_args.dim, bias=model_args.bias)

        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.resid_dropout = nn.Dropout(model_args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq_len, dim

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)

        # Reshape for multi-head attention: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def init_weights(self, init_std: float):
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=init_std)
        if self.c_attn.bias is not None:
            nn.init.zeros_(self.c_attn.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)


class MLP(nn.Module):
    """Feed-forward network with GELU activation (GPT-2 style)."""

    def __init__(self, model_args: GPT2ModelArgs):
        super().__init__()
        hidden_dim = 4 * model_args.dim  # GPT-2 uses 4x expansion

        self.c_fc = nn.Linear(model_args.dim, hidden_dim, bias=model_args.bias)
        self.c_proj = nn.Linear(hidden_dim, model_args.dim, bias=model_args.bias)
        self.dropout = nn.Dropout(model_args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def init_weights(self, init_std: float):
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=init_std)
        if self.c_fc.bias is not None:
            nn.init.zeros_(self.c_fc.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)


class TransformerBlock(nn.Module):
    """GPT-2 transformer block with pre-LayerNorm."""

    def __init__(self, layer_id: int, model_args: GPT2ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.ln_1 = nn.LayerNorm(model_args.dim, eps=model_args.norm_eps, bias=model_args.bias)
        self.attn = CausalSelfAttention(model_args)
        self.ln_2 = nn.LayerNorm(model_args.dim, eps=model_args.norm_eps, bias=model_args.bias)
        self.mlp = MLP(model_args)

        # For depth-scaled initialization
        self.n_layers = model_args.n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm architecture (GPT-2 style)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def init_weights(self, buffer_device: torch.device | None = None):
        # GPT-2 uses 0.02 std for most layers, scaled by 1/sqrt(2*n_layers) for residual
        init_std = 0.02
        residual_std = init_std / math.sqrt(2 * self.n_layers)

        self.ln_1.reset_parameters()
        self.ln_2.reset_parameters()
        self.attn.init_weights(init_std)
        self.mlp.init_weights(init_std)

        # Scale residual projections
        nn.init.normal_(self.attn.c_proj.weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.mlp.c_proj.weight, mean=0.0, std=residual_std)


class GPT2Model(nn.Module, ModelProtocol):
    """
    GPT-2 Language Model.

    Architecture:
    - Token embeddings + learned positional embeddings
    - N transformer blocks with pre-LayerNorm
    - Final LayerNorm + linear output (optionally weight-tied)
    """

    def __init__(self, model_args: GPT2ModelArgs):
        super().__init__()
        self.model_args = model_args

        # Token and position embeddings
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.pos_embeddings = nn.Embedding(model_args.max_seq_len, model_args.dim)
        self.drop = nn.Dropout(model_args.dropout)

        # Transformer blocks stored in ModuleDict for compatibility with TorchTitan
        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        # Final layer norm
        self.norm = nn.LayerNorm(model_args.dim, eps=model_args.norm_eps, bias=model_args.bias)

        # Output projection (may be weight-tied with tok_embeddings)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # Weight tying
        if model_args.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        # Register position indices as buffer
        self.register_buffer(
            "position_ids",
            torch.arange(model_args.max_seq_len).unsqueeze(0),
            persistent=False
        )

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for GPT-2 with optional fused loss computation.

        When `labels` is provided, computes and returns the loss directly.
        This enables torch.compile to fuse the output projection with cross-entropy,
        avoiding materialization of the full [batch, seq, vocab_size] logits tensor.

        Memory savings from fused loss:
        - Without fusion: logits tensor = batch × seq × vocab × dtype_size
          For GPT-2 (vocab=50257): 64 × 1024 × 50257 × 2 bytes = ~6.5GB in bf16
        - With fusion: Only chunked computation, ~50% memory reduction

        Args:
            tokens: Input token IDs, shape (batch_size, seq_len)
            labels: Target token IDs for loss computation, shape (batch_size, seq_len).
                    If None, returns logits for inference.

        Returns:
            If labels is None: Logits, shape (batch_size, seq_len, vocab_size)
            If labels is provided: Scalar loss value
        """
        B, T = tokens.size()
        assert T <= self.model_args.max_seq_len, f"Sequence length {T} exceeds max {self.model_args.max_seq_len}"

        # Get embeddings
        tok_emb = self.tok_embeddings(tokens)  # (B, T, dim)
        pos_emb = self.pos_embeddings(self.position_ids[:, :T])  # (1, T, dim)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for layer in self.layers.values():
            x = layer(x)

        # Final norm
        x = self.norm(x)

        if labels is not None:
            # FUSED PATH: Compute loss without materializing full logits tensor.
            # When compiled, torch.compile can fuse the linear projection with
            # cross-entropy, using chunked/online computation similar to Flash Attention.
            # This avoids storing the [batch, seq, vocab_size] logits tensor.
            logits = self.output(x)
            # Flatten for cross-entropy: (B, T, V) -> (B*T, V) and (B, T) -> (B*T,)
            loss = F.cross_entropy(
                logits.flatten(0, 1).float(),
                labels.flatten(0, 1),
            )
            return loss
        else:
            # INFERENCE PATH: Return logits for generation/evaluation
            logits = self.output(x)
            return logits

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights following GPT-2 conventions."""
        init_std = 0.02

        # Initialize embeddings
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=init_std)

        # Initialize transformer blocks
        for layer in self.layers.values():
            layer.init_weights(buffer_device)

        # Initialize final layer norm
        self.norm.reset_parameters()

        # Initialize output (if not weight-tied)
        if not self.model_args.weight_tying:
            nn.init.normal_(self.output.weight, mean=0.0, std=init_std)

        # Reinitialize position buffer on correct device
        if buffer_device is not None:
            self.position_ids = torch.arange(
                self.model_args.max_seq_len, device=buffer_device
            ).unsqueeze(0)
