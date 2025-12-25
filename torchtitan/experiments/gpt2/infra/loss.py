"""
GPT-2 Fused Loss Function
=========================

This module provides a loss function builder for GPT-2 that supports fused
forward+loss computation when the model computes loss internally.

FUSED vs SEPARATE LOSS COMPUTATION:
-----------------------------------
Traditional approach (TorchTitan default):
    logits = model(inputs)           # Materializes [batch, seq, vocab] tensor
    loss = cross_entropy(logits, labels)  # Compiled separately

Fused approach (this implementation):
    loss = model(inputs, labels=labels)  # Loss computed inside forward
    return loss  # Pass-through, loss already computed

The fused approach enables torch.compile to optimize the output projection
and cross-entropy together, potentially using chunked computation to avoid
materializing the full logits tensor.

MEMORY SAVINGS:
--------------
For GPT-2 with vocab_size=50257, batch=64, seq=1024:
- Logits tensor: 64 × 1024 × 50257 × 2 bytes (bf16) = ~6.5GB
- With fusion: Chunked computation, ~50% memory reduction

See: https://github.com/pytorch/torchtune/pull/2507
"""

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def fused_cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Loss function for fused forward+loss computation.

    When the model computes loss internally (pred is already a scalar loss),
    this function simply returns it. When the model returns logits (inference
    mode or non-fused path), this computes cross-entropy normally.

    Args:
        pred: Either:
            - Scalar loss tensor (fused path, model computed loss)
            - Logits tensor of shape (batch, seq, vocab) (inference path)
        labels: Target token IDs of shape (batch, seq)

    Returns:
        Scalar loss tensor
    """
    # Check if pred is already a scalar loss (fused path)
    if pred.dim() == 0 or (pred.dim() == 1 and pred.numel() == 1):
        # Model already computed the loss, just return it
        return pred.squeeze()

    # Fallback: pred is logits, compute cross-entropy
    # This path is used for inference/validation or if forward wasn't called with labels
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_fused_cross_entropy_loss(job_config: JobConfig, **kwargs):
    """
    Build a loss function for GPT-2 with fused forward+loss support.

    Unlike the standard TorchTitan loss builder, this does NOT compile the loss
    function separately. Instead, the loss computation is inside the model's
    forward pass, which is compiled as a whole. This enables:

    1. Output projection + cross-entropy fusion
    2. Avoidance of full logits tensor materialization
    3. Better memory efficiency for large vocabulary models

    The returned function handles both:
    - Fused path: pred is already the loss (pass-through)
    - Inference path: pred is logits (compute cross-entropy)

    Args:
        job_config: Job configuration (used for consistency with TorchTitan API)
        **kwargs: Additional arguments (ignored)

    Returns:
        Loss function that handles both fused and non-fused cases
    """
    del kwargs  # Unused, but kept for API compatibility

    # NOTE: We intentionally do NOT compile the loss function here.
    # The loss computation is inside model.forward() and will be compiled
    # together with the model for optimal fusion.
    #
    # If we compiled this separately, we'd lose the fusion opportunity
    # because torch.compile boundaries prevent cross-function optimization.

    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info(
            "GPT-2 uses fused forward+loss compilation. "
            "Loss is computed inside model.forward() and compiled together with the model. "
            "Separate loss compilation is skipped for optimal fusion."
        )

    return fused_cross_entropy_loss
