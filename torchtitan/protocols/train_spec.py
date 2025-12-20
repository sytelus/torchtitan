# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TrainSpec - Model-Specific Training Configuration
==================================================

This module defines TrainSpec, the core abstraction that allows TorchTitan's
generic Trainer to work with different model architectures (Llama, DeepSeek,
Flux, etc.) without architecture-specific code in the training loop.

DESIGN PHILOSOPHY:
------------------
TorchTitan separates "what" to train from "how" to train:
- **Trainer**: Generic training loop (load data, forward, backward, optimize)
- **TrainSpec**: Model-specific components (model class, parallelization, etc.)

This allows:
1. Adding new models without modifying the Trainer
2. Reusing the same Trainer for different architectures
3. Customizing any training component per-model

TRAINSPEC COMPONENTS:
---------------------
1. **model_cls**: The model class (e.g., Transformer)
2. **model_args**: Model configurations by flavor (e.g., "8B", "70B")
3. **parallelize_fn**: How to apply TP, AC, compile, FSDP to this model
4. **pipelining_fn**: How to split model for Pipeline Parallelism (optional)
5. **build_optimizers_fn**: Create optimizer(s) for the model
6. **build_lr_schedulers_fn**: Create LR scheduler(s)
7. **build_dataloader_fn**: Create the data loader
8. **build_tokenizer_fn**: Create tokenizer (optional, not needed for images)
9. **build_loss_fn**: Create the loss function
10. **build_validator_fn**: Create validation evaluator (optional)
11. **build_metrics_processor_fn**: Create metrics logger (optional)
12. **state_dict_adapter**: Convert state dict formats (e.g., for HuggingFace)

ADDING A NEW MODEL:
-------------------
To add support for a new model architecture:

1. Create model directory: `torchtitan/models/my_model/`
2. Implement model class in `model.py`
3. Implement parallelization in `parallelize.py`
4. Create `__init__.py` with `get_train_spec()`:

```python
def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=MyModel,
        model_args={"8B": MyModelArgs_8B, ...},
        parallelize_fn=parallelize_my_model,
        pipelining_fn=None,  # or pipeline_my_model
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
```

5. Register in `torchtitan/models/__init__.py`:
   ```python
   _supported_models = [..., "my_model"]
   ```

CUSTOM TRAINSPECS:
------------------
For external models, use `register_train_spec()`:

```python
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

my_spec = TrainSpec(...)
register_train_spec("my_custom_model", my_spec)
```

Then set `--model.name=my_custom_model` in the config.

TRAINSPEC LOOKUP ORDER:
-----------------------
1. User-registered specs (via register_train_spec)
2. Built-in models (torchtitan/models/)
3. Experimental models (torchtitan/experiments/)
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Mapping, TypeAlias

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import BaseValidator
from torchtitan.config import LRScheduler

from .model import BaseModelArgs, ModelProtocol
from .state_dict_adapter import BaseStateDictAdapter


# =============================================================================
# TYPE ALIASES
# These define the expected signatures for TrainSpec builder functions
# =============================================================================

# Function that applies parallelization (TP, AC, compile, FSDP) to model
ParallelizeFunction: TypeAlias = Callable[..., nn.Module]

# Function that sets up Pipeline Parallelism
# Returns: (schedule, model_parts, has_first_stage, has_last_stage)
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]

# Builders for various training components
DataLoaderBuilder: TypeAlias = Callable[..., BaseDataLoader]
TokenizerBuilder: TypeAlias = Callable[..., BaseTokenizer]
MetricsProcessorBuilder: TypeAlias = Callable[..., MetricsProcessor]
OptimizersBuilder: TypeAlias = Callable[..., OptimizersContainer]
LRSchedulersBuilder: TypeAlias = Callable[
    [OptimizersContainer, LRScheduler, int], LRSchedulersContainer
]
LossFunctionBuilder: TypeAlias = Callable[..., LossFunction]
ValidatorBuilder: TypeAlias = Callable[..., BaseValidator]


@dataclass
class TrainSpec:
    """
    Bundle of model-specific builders used by the generic Trainer.

    TrainSpec encapsulates all model-specific components needed for training,
    allowing the Trainer to remain architecture-agnostic. Each model (Llama,
    DeepSeek, Flux, etc.) provides its own TrainSpec implementation.

    Attributes:
        model_cls (type[ModelProtocol]): The model class to instantiate.
            Must implement forward() and init_weights() methods.

        model_args (Mapping[str, BaseModelArgs]): Model configurations by
            "flavor" name (e.g., "8B", "70B", "405B"). Contains hyperparameters
            like hidden_dim, n_layers, n_heads, etc.

        parallelize_fn (ParallelizeFunction): Function to apply parallelization
            strategies to the model. Typically applies TP -> AC -> compile -> FSDP.

        pipelining_fn (PipeliningFunction | None): Function to set up Pipeline
            Parallelism. Returns the PP schedule and model chunks. None if PP
            is not supported for this model.

        build_optimizers_fn (OptimizersBuilder): Factory for creating optimizer(s).
            Creates OptimizersContainer that handles multiple optimizers for PP.

        build_lr_schedulers_fn (LRSchedulersBuilder): Factory for creating LR
            scheduler(s). Matches optimizers for PP compatibility.

        build_dataloader_fn (DataLoaderBuilder): Factory for creating the data
            loader. Handles data sharding for DP.

        build_tokenizer_fn (TokenizerBuilder | None): Factory for creating the
            tokenizer. None for non-text models (e.g., image generation).

        build_loss_fn (LossFunctionBuilder): Factory for creating the loss
            function. May be wrapped for loss parallel or gradient accumulation.

        build_validator_fn (ValidatorBuilder | None): Optional factory for
            validation evaluator. Enables periodic validation during training.

        build_metrics_processor_fn (MetricsProcessorBuilder | None): Optional
            custom metrics processor. Falls back to default if None.

        state_dict_adapter (type[BaseStateDictAdapter] | None): Optional adapter
            for converting between native and external state dict formats
            (e.g., HuggingFace). Enables loading/saving HF checkpoints.

    Example:
        ```python
        spec = TrainSpec(
            model_cls=Transformer,
            model_args={"8B": TransformerArgs(dim=4096, n_layers=32, ...)},
            parallelize_fn=parallelize_transformer,
            pipelining_fn=pipeline_transformer,
            build_optimizers_fn=build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=build_hf_dataloader,
            build_tokenizer_fn=build_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
        )
        ```
    """

    model_cls: type[ModelProtocol]
    model_args: Mapping[str, BaseModelArgs]
    parallelize_fn: ParallelizeFunction
    pipelining_fn: PipeliningFunction | None
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_dataloader_fn: DataLoaderBuilder
    build_tokenizer_fn: TokenizerBuilder | None
    build_loss_fn: LossFunctionBuilder
    build_validator_fn: ValidatorBuilder | None = None
    build_metrics_processor_fn: MetricsProcessorBuilder | None = None
    state_dict_adapter: type[BaseStateDictAdapter] | None = None


_extra_train_specs: dict[str, TrainSpec] = {}


def register_train_spec(name: str, train_spec: TrainSpec) -> None:
    global _extra_train_specs
    if name in _extra_train_specs:
        raise ValueError(f"TrainSpec {name} is already registered.")

    # user can define a TrainSpec from outside of torchtitan
    _extra_train_specs[name] = train_spec


def get_train_spec(name: str) -> TrainSpec:
    # user-defined TrainSpec has higher priority
    global _extra_train_specs
    if name in _extra_train_specs:
        return _extra_train_specs[name]

    from torchtitan.experiments import _supported_experiments
    from torchtitan.models import _supported_models

    if name in _supported_models:
        module = import_module(f"torchtitan.models.{name}")
        return module.get_train_spec()
    elif name in _supported_experiments:
        module = import_module(f"torchtitan.experiments.{name}")
        return module.get_train_spec()

    raise ValueError(f"TrainSpec {name} is not registered.")
