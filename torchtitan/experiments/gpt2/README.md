# GPT-2 Experiment

A simple GPT-2 implementation for learning TorchTitan basics.

## Overview

This experiment provides a clean GPT-2 implementation that demonstrates:
- How to add a new model to TorchTitan
- Single GPU training without parallelism
- Multi-GPU training with DDP (and optional FSDP)
- Integration with HuggingFace datasets

## Supported Model Sizes

| Flavor | Parameters | Layers | Dim | Heads |
|--------|------------|--------|-----|-------|
| tiny | ~49M | 6 | 384 | 6 |
| 124M | 124M | 12 | 768 | 12 |

## Quick Start

### 1. Download GPT-2 Tokenizer

If you have `tiktoken` installed, you can skip this step and use the default
tokenizer (no downloads required).

```bash
python scripts/download_hf_assets.py \
    --repo_id openai-community/gpt2 \
    --assets tokenizer \
    --local_dir ./assets/hf/gpt2
```

### 2. Single GPU Tiny Training (Shakespeare)

```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/debug_model.toml" ./run_train.sh
```

### 3. 8 GPU DDP Training (OpenWebText)

```bash
CONFIG_FILE="./torchtitan/experiments/gpt2/train_configs/gpt2_124m_openwebtext.toml" ./run_train.sh
```

## Datasets

- **tiny_shakespeare**: 40k lines of Shakespeare for debugging
- **openwebtext**: First 5GB of OpenWebText (~8M samples)

## Limitations

This tutorial implementation does NOT support:
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Context Parallelism (CP)

For advanced parallelism, see the Llama3 model implementation.

## Architecture Differences from Llama

| Feature | GPT-2 | Llama |
|---------|-------|-------|
| Positional Encoding | Learned | RoPE |
| Activation | GELU | SwiGLU |
| Normalization | LayerNorm | RMSNorm |
| Weight Tying | Optional | No |
| GQA | No | Yes |
