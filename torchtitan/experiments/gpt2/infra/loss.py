"""
GPT-2 Fused Loss Function
=========================

This module provides a loss function builder for GPT-2 that supports fused
forward+loss computation when the model computes loss internally.

GPT-2 MODEL RETURN FORMAT:
--------------------------
The GPT-2 model.forward() returns a tuple: (logits, loss)
- logits: Always returned, shape (batch, seq, vocab)
- loss: Scalar if labels were provided, else None

FUSED vs SEPARATE LOSS COMPUTATION:
-----------------------------------
Traditional approach (TorchTitan default):
    logits = model(inputs)           # Materializes [batch, seq, vocab] tensor
    loss = cross_entropy(logits, labels)  # Compiled separately

Fused approach (this implementation):
    logits, loss = model(inputs, labels=labels)  # Loss computed inside forward
    return loss  # Use pre-computed loss

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


def fused_cross_entropy_loss(
    pred: tuple[torch.Tensor, torch.Tensor | None],
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Loss function for fused forward+loss computation.

    The GPT-2 model returns (logits, loss) tuple. If loss was computed inside
    the model (labels were provided), we use that. Otherwise, we compute
    cross-entropy from the logits.

    Args:
        pred: Tuple of (logits, loss) from model.forward():
            - logits: Shape (batch, seq, vocab)
            - loss: Scalar if labels provided to forward(), else None
        labels: Target token IDs of shape (batch, seq)

    Returns:
        Scalar loss tensor
    """
    logits, loss = pred

    # If model already computed the loss, use it
    if loss is not None:
        return loss

    # Fallback: compute cross-entropy from logits
    # This path is used for inference/validation or if forward wasn't called with labels
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), labels.flatten(0, 1)
    )


# Signal to the trainer that this loss expects labels to be passed into model.forward().
# When set, the trainer will include labels in the forward() call, enabling the model
# to compute loss internally. This allows torch.compile to fuse output projection
# with cross-entropy, avoiding full logits materialization.
#
# How it works:
#   1. Trainer sees `loss_fn.requires_labels_in_forward = True`
#   2. Trainer calls `model(inputs, labels=labels)` instead of `model(inputs)`
#   3. Model computes loss inside forward() and returns `(logits, loss)`
#   4. This loss function extracts the pre-computed loss from the tuple
#
# See: torchtitan/train.py (training path) and components/validate.py (validation path)
fused_cross_entropy_loss.requires_labels_in_forward = True


def build_fused_cross_entropy_loss(job_config: JobConfig, **kwargs):
    """
    Build a loss function for GPT-2 with fused forward+loss support.

    Unlike the standard TorchTitan loss builder, this does NOT compile the loss
    function separately. Instead, the loss computation is inside the model's
    forward pass, which is compiled as a whole. This enables:

    1. Output projection + cross-entropy fusion
    2. Avoidance of full logits tensor materialization
    3. Better memory efficiency for large vocabulary models

    The returned function handles:
    - Fused path: loss from model tuple is used directly
    - Fallback path: compute cross-entropy from logits if loss is None

    Args:
        job_config: Job configuration (used for consistency with TorchTitan API)
        **kwargs: Additional arguments (ignored)

    Returns:
        Loss function that handles GPT-2's (logits, loss) tuple output
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
