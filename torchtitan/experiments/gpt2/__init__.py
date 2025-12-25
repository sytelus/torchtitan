"""
GPT-2 Experiment for TorchTitan
===============================

This experiment provides a simple GPT-2 implementation for learning TorchTitan.
It supports:
- GPT-2 124M (debugmodel, default)
- GPT-2 355M
- GPT-2 774M
- GPT-2 1.5B

COMPILATION STRATEGY:
This implementation uses WHOLE-MODEL compilation with FUSED FORWARD+LOSS:
- The model computes loss inside forward() when labels are provided
- torch.compile wraps the entire model (not per-layer)
- This enables fusion of output projection + cross-entropy, avoiding
  materialization of the full [batch, seq, vocab_size] logits tensor
- Memory savings: ~50% reduction for large vocabulary models

See infra/parallelize.py and infra/loss.py for implementation details.

Usage:
    CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/debug_model.toml" ./run_train.sh
"""

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.loss import build_fused_cross_entropy_loss
from .infra.parallelize import parallelize_gpt2
from .model.args import GPT2ModelArgs
from .model.model import GPT2Model
from .model.state_dict_adapter import GPT2StateDictAdapter

__all__ = [
    "parallelize_gpt2",
    "GPT2ModelArgs",
    "GPT2Model",
    "GPT2StateDictAdapter",
    "gpt2_configs",
]


# GPT-2 model configurations
gpt2_configs = {
    # Debug model for testing (very small)
    "debugmodel": GPT2ModelArgs(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    # GPT-2 124M (Small)
    "124M": GPT2ModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    # GPT-2 355M (Medium)
    "355M": GPT2ModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=16,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    # GPT-2 774M (Large)
    "774M": GPT2ModelArgs(
        dim=1280,
        n_layers=36,
        n_heads=20,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    # GPT-2 1.5B (XL)
    "1558M": GPT2ModelArgs(
        dim=1600,
        n_layers=48,
        n_heads=25,
        vocab_size=50257,
        max_seq_len=1024,
    ),
}


def get_train_spec() -> TrainSpec:
    """Return the TrainSpec for GPT-2 models.

    This TrainSpec uses a fused forward+loss compilation strategy:
    - The model computes loss inside forward() when labels are provided
    - Whole-model compilation enables output+cross_entropy fusion
    - The loss builder returns a pass-through function since loss is pre-computed
    """
    return TrainSpec(
        model_cls=GPT2Model,
        model_args=gpt2_configs,
        parallelize_fn=parallelize_gpt2,
        pipelining_fn=None,  # Pipeline parallelism not implemented
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_fused_cross_entropy_loss,  # Fused loss for optimal compilation
        state_dict_adapter=GPT2StateDictAdapter,
    )
