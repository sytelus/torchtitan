# Installation and Version Requirements

This document covers the PyTorch, CUDA, and other version requirements for
running TorchTitan.

## Version Summary

| Component | Version | Notes |
|-----------|---------|-------|
| **PyTorch** | Nightly (latest) | Required for latest features |
| **CUDA** | 12.4, 12.6, 12.8 | CI uses 12.6; Docker base uses 12.4 |
| **ROCm** | 7.0 | For AMD GPUs |
| **Python** | 3.10+ (3.12 recommended) | CI uses Python 3.12 |
| **TorchAO** | Nightly (latest) | Required for FP8/quantization features |

## Why PyTorch Nightly?

TorchTitan is under active development and relies on the latest PyTorch
features that may not yet be in stable releases:

- FSDP2 improvements
- Async tensor parallelism
- FP8/MXFP8 training support
- Distributed checkpointing enhancements
- torch.compile optimizations

> "To use the latest features of `torchtitan`, we recommend using the most
> recent PyTorch nightly." — README.md

---

## Installation Options

### Option 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/pytorch/torchtitan
cd torchtitan

# Install PyTorch nightly (CUDA 12.6)
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

# Install TorchAO nightly (for FP8 support)
USE_CPP=0 pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126

# Install TorchTitan and dependencies
pip install -e .
```

### Option 2: Nightly Build

```bash
# Install PyTorch nightly
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

# Install TorchTitan nightly
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Option 3: Stable Release

```bash
# Install from PyPI
pip install torchtitan

# Or from conda
conda install torchtitan
```

Note: Stable releases pin specific nightly versions of `torch` and `torchao`.
Check the [release notes](https://github.com/pytorch/torchtitan/releases) for
the exact versions.

---

## CUDA Version Options

TorchTitan supports multiple CUDA versions. Replace `cu126` in the install
commands with your preferred version:

| CUDA Version | Index URL Suffix | Notes |
|--------------|------------------|-------|
| CUDA 12.6 | `cu126` | Default in CI |
| CUDA 12.8 | `cu128` | Latest |
| CUDA 12.4 | `cu124` | Docker base image |

**Example for CUDA 12.8:**
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## AMD GPU (ROCm) Support

TorchTitan supports AMD GPUs via ROCm:

```bash
# ROCm 7.0
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.0 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

---

## Hardware Requirements by Feature

| Feature | Minimum GPU | Notes |
|---------|-------------|-------|
| Basic training | Any CUDA GPU | FSDP, TP, PP all work |
| FP8 training | H100 (SM89+) | Requires tensor cores with FP8 support |
| MXFP8 training | B200 (SM100+) | Blackwell architecture only |
| Async TP | H100+ recommended | Benefits from NVLink |

---

## CI Docker Images

TorchTitan CI uses custom Docker images based on NVIDIA's CUDA images:

### CUDA Image
```
Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04
Python: 3.12
Clang: 12
```

### ROCm Image
```
Base: rocm/dev-ubuntu-22.04:latest
Python: 3.12
Clang: 12
```

### Building the Docker Image

```bash
cd .ci/docker

# Build CUDA image
./build.sh torchtitan-ubuntu-20.04-clang12 -t torchtitan:cuda

# Build ROCm image
./build.sh torchtitan-rocm-ubuntu-22.04-clang12 -t torchtitan:rocm
```

---

## Verifying Installation

After installation, verify your setup:

```bash
# Check PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Check TorchTitan
python -c "import torchtitan; print('TorchTitan installed successfully')"

# Check TorchAO (for FP8)
python -c "import torchao; print(f'TorchAO: {torchao.__version__}')"

# Run a quick validation (single GPU, fake backend)
NGPU=1 COMM_MODE=fake_backend python -m torchtitan.train \
    --job.config_file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --training.steps 1
```

---

## Tested Version Combinations

These version combinations have been validated in benchmarks:

| Date | PyTorch | TorchAO | Hardware | Benchmark |
|------|---------|---------|----------|-----------|
| 2025/06 | `2.8.0a0+5228986c39` | `0afa4c1` | H200 | Llama3-8B |
| 2025/06 | `38410cf9` | `6243040` | H100 | Async TP |
| 2024/12 | `1963fc8` | `eab345c` | H100 | Llama3 baseline |

---

## Troubleshooting

### CUDA Version Mismatch

If you see CUDA version errors:
```bash
# Check your CUDA driver version
nvidia-smi

# Ensure PyTorch CUDA version matches or is lower than driver version
python -c "import torch; print(torch.version.cuda)"
```

### ImportError for TorchAO

If FP8 features fail to import:
```bash
# Reinstall TorchAO from source
USE_CPP=0 pip install git+https://github.com/pytorch/ao.git
```

### ROCm-specific Issues

For AMD GPUs, ensure ROCm is properly installed:
```bash
# Check ROCm version
rocm-smi --version

# Verify PyTorch can see ROCm GPUs
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dependencies

Core dependencies (from `requirements.txt`):

```
torchdata >= 0.8.0
datasets >= 3.6.0
tomli >= 1.1.0  # Python < 3.11 only
tensorboard
tabulate
wandb
fsspec
tyro
tokenizers >= 0.15.0
safetensors
psutil
einops
pillow
```

For specific features:
- **FP8 training**: `torchao` (nightly)
- **Fault tolerance**: `torchft-nightly`
- **Flux models**: See `.ci/docker/requirements-flux.txt`
- **VLM models**: See `.ci/docker/requirements-vlm.txt`
