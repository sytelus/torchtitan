To support rapid experimentation with torchtitan, we provide several extension points. The principle for adding these extension points is to support various use cases with flexible component swapping and reuse, while trying to keep the code clean and minimal.

The extension points and protocols mentioned in this note are subject to change.


### `TrainSpec`

[`TrainSpec`](../torchtitan/protocols/train_spec.py) supports configuring high-level components in model training, including
- definitions of model class and model args
- model parallelization functions
- loss functions
- factory methods for creating dataloader / tokenizer / optimizer / learning rate scheduler / metrics processor

The coarse level abstraction tries to hit a balance between flexible component swapping and a straightforward train script ([train.py](../torchtitan/train.py)).
Note that among all training components, currently [`CheckpointManager`](../torchtitan/components/checkpoint.py) and [`FTManager`](../torchtitan/components/ft/manager.py) are not configurable since we do not expect them to be customized, but we are open to requests.

To register a `TrainSpec`, please use the `register_train_spec` API, and make sure registration happens before `get_train_spec` is called during training initialization. In torchtitan, `get_train_spec` will dynamically look for models in `torchtitan/models` or `torchtitan/experiments`.


### `ModelConverter`

Originated from a [request](https://github.com/pytorch/torchtitan/issues/790) to unify quantization interface and supports dynamic registration,
[`ModelConverter`](../torchtitan/protocols/model_converter.py) defines the following general interface:
- `convert` is called after model definition and meta device initialization, but before model parallelization. It can perform general module rewrite, e.g. [Float8](../torchtitan/components/quantization/float8.py) module swapping, as long as it is compatible with other components.
- `post_optimizer_hook`, as its name suggests, would be registered (via `torch.optim.Optimizer.register_step_post_hook`) to perform necessary post optimizer step operations. As an example, the [Float8](../torchtitan/components/quantization/float8.py) component in torchtitan uses this hook to issue a single all-reduce for all FSDP2 parameters (at once for better performance) to calculate the dynamic scale.

To register a `ModelConverter`, please follow the example of [Float8](../torchtitan/components/quantization/float8.py) to `register_model_converter`. Please make sure the registration code is called before training initialization. In torchtitan, it is performed during  [module import](../torchtitan/__init__.py).

#### Why Model Converters Exist

Model converters are a plugin system that transforms the model structure **before** parallelization is applied. They serve several critical use cases:

1. **FP8 Training**: Replace `nn.Linear` with `Float8Linear` that performs 8-bit matrix multiplications
2. **Fused Layers**: Replace standard attention with flash attention variants
3. **Quantization-Aware Training (QAT)**: Modify layers to simulate quantization during training

#### Why Converters Run Before Parallelization

The timing is critical because:
- Converters may add new parameters (e.g., FP8 scale tensors)
- TP/FSDP parallelization relies on the final module structure
- Once FSDP wraps modules, structure changes become complex

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Model Converter Timing                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Model Definition    Model Converters      Parallelization         │
│   (meta device)           apply            (TP, FSDP, PP)           │
│        │                    │                    │                  │
│        ▼                    ▼                    ▼                  │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐            │
│   │ Create  │   ───►   │ Swap    │   ───►   │ Shard   │            │
│   │ model   │          │ modules │          │ params  │            │
│   │ on meta │          │ (FP8,   │          │ (FSDP), │            │
│   │ device  │          │  QAT)   │          │ split   │            │
│   └─────────┘          └─────────┘          │ tensors │            │
│                                             │ (TP)    │            │
│                                             └─────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### The `post_optimizer_hook` Explained

For FP8 training, after each optimizer step, we need to recompute dynamic scaling factors (`amax`) for all parameters. The hook allows batch computation across all parameters with a single all-reduce, which is more efficient than per-layer scaling:

```python
# During training initialization (train.py)
model_converters = build_model_converters(job_config, parallel_dims)
model_converters.convert(model)  # In-place modification BEFORE parallelization

# ... later, after optimizer step (via registered hook):
model_converters.post_optimizer_hook(model_parts)  # Batch amax computation
```


### Train script

To perform various tasks, from adding a new model (possibly with a new modality), to trying out a new training paradigm (e.g. async training), a single train script cannot handle all the cases, unless customization points are inserted everywhere to make it less readable. Instead of always starting and maintaining a standalone train script, we group code in [train.py](../torchtitan/train.py) into functions to allow for reuse.

This is an ongoing effort, and the level of grouping is subject to change.


### Extending `JobConfig`

[`JobConfig`](../torchtitan/config/job_config.py) supports custom extension through the `--job.custom_config_module` flag.
This lets you define a custom module that extends `JobConfig` with additional fields.

When specified, your custom `JobConfig` is merged with the default:
- If a field exists in both, the custom config’s value replaces the default.
- Fields unique to either config are retained.

#### Example

To add a custom `custom_config` section, define your own `JobConfig`:

```python
# torchtitan/experiments/your_folder/job_config.py
from dataclasses import dataclass, field

@dataclass
class CustomConfig:
    how_is_your_day: str = "good"
    """Just an example."""

@dataclass
class Training:
    steps: int = 500
    """Replaces the default value"""

    my_mini_steps: int = 10000
    """New field is added"""

    ... # Original fields are preserved

@dataclass
class JobConfig:
    custom_config: CustomConfig = field(default_factory=CustomConfig)
    training: Training= field(default_factory=Training)
```

Then run your script with:

```bash
--job.custom_config_module=torchtitan.experiments.your_folder.job_config
```

Or specify it in your `.toml` config:

```toml
[job]
custom_config_module = "torchtitan.experiments.your_folder.job_config"
```


### Learning Rate Scheduler Customization

TorchTitan uses `LambdaLR` via `LRSchedulersContainer` with three decay types:

#### Available Schedules

```toml
[lr_scheduler]
warmup_steps = 2000        # Linear warmup steps
decay_ratio = 0.8          # When to start decay (0.8 = last 80% of training)
decay_type = "cosine"      # "linear", "sqrt", or "cosine"
min_lr_factor = 0.1        # Don't go below 10% of base LR
```

**Schedule patterns:**
- **Warmup-Decay (WD)**: Set `decay_ratio = null` - decay starts immediately after warmup
- **Warmup-Stable-Decay (WSD)**: Set `decay_ratio = 0.8` - stable LR after warmup, decay in last 80%

#### Custom LR Scheduler

Create a custom scheduler by overriding `build_lr_schedulers_fn` in your TrainSpec:

```python
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import LRScheduler as LRSchedulerConfig
import functools

def build_custom_lr_schedulers(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> LRSchedulersContainer:
    """Custom scheduler with your own schedule logic."""

    def my_schedule(current_step, warmup_steps, total_steps):
        if current_step < warmup_steps:
            return float(current_step) / warmup_steps  # Linear warmup
        # Your custom decay logic
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.1, 1.0 - progress * 0.9)  # Linear decay to 10%

    lr_lambda = functools.partial(
        my_schedule,
        warmup_steps=lr_scheduler_config.warmup_steps,
        total_steps=training_steps,
    )

    return LRSchedulersContainer(optimizers, lr_lambda)

# Register in TrainSpec
train_spec = TrainSpec(
    ...
    build_lr_schedulers_fn=build_custom_lr_schedulers,
)
```


### Custom Optimizers

TorchTitan currently supports **Adam** and **AdamW** only. To add custom optimizers
(e.g., Muon, LAMB, Sophia), modify the `build_optimizers` function or create a
custom TrainSpec.

#### Current Optimizer Configuration

```toml
[optimizer]
name = "AdamW"              # "Adam" or "AdamW"
lr = 8e-4
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
weight_decay = 0.1
implementation = "fused"    # "fused", "foreach", or "for-loop"
```

#### Adding a Custom Optimizer (e.g., Muon)

Muon and other optimizers are **not currently supported**. To add one:

**Option 1: Modify optimizer.py directly**

```python
# In torchtitan/components/optimizer.py, add to optimizer_classes dict:
optimizer_classes = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "Muon": muon.Muon,  # Add import and class
}
```

**Option 2: Create custom TrainSpec (recommended)**

```python
from torchtitan.protocols.train_spec import TrainSpec
from torchtitan.components.optimizer import OptimizersContainer
import muon  # Your optimizer library

def build_custom_optimizers(model_parts, optimizer_config, parallel_dims, ft_manager=None):
    """Build Muon optimizer for all model parts."""
    optimizers = []
    for model in model_parts:
        opt = muon.Muon(
            model.parameters(),
            lr=optimizer_config.lr,
            momentum=optimizer_config.beta1,
        )
        optimizers.append(opt)
    return OptimizersContainer(optimizers, ...)

train_spec = TrainSpec(
    ...
    build_optimizers_fn=build_custom_optimizers,
)
```

**Note:** Custom optimizers may require additional handling for distributed training
(FSDP sharding, gradient scaling, etc.).
