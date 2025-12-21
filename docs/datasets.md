# Custom Datasets in torchtitan

`torchtitan` is designed to work seamlessly with most HuggingFace datasets. While we provide the C4 dataset for numerics and convergence testing, you can easily add support for your own datasets.

## Quick Start

Locate the dataset configuration file:
```
torchtitan/hf_datasets/text_datasets.py
```

The `DatasetConfig` dataclass is defined in:
```
torchtitan/hf_datasets/__init__.py
```

---

## Tokenization Architecture

TorchTitan tokenizes text **on-the-fly during training**, not as a preprocessing step. This provides memory efficiency and flexibility.

### Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tokenization Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐ │
│  │ HuggingFace      │     │ HuggingFaceText     │     │ DataLoader   │ │
│  │ Dataset          │────►│ Dataset             │────►│              │ │
│  │ (raw text)       │     │ (tokenizes on iter) │     │ (batches)    │ │
│  └──────────────────┘     └─────────────────────┘     └──────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│                           ┌─────────────────────┐                       │
│                           │ HuggingFaceTokenizer│                       │
│                           │ (wraps HF tokenizers│                       │
│                           │  library)           │                       │
│                           └─────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. HuggingFaceTokenizer (`torchtitan/components/tokenizer.py`)

A wrapper around HuggingFace's `tokenizers` library that handles:
- Loading tokenizer files from `hf_assets_path`
- Inferring BOS/EOS tokens from `tokenizer_config.json`
- Intelligent encoding with optional BOS/EOS addition

**Supported tokenizer formats:**
| Format | Files Required | Notes |
|--------|----------------|-------|
| Modern (preferred) | `tokenizer.json` | Full tokenizer definition |
| BPE | `vocab.json` + `merges.txt` | GPT-2 style tokenizers |
| WordPiece | `vocab.txt` | BERT-style tokenizers |

**Key methods:**
```python
tokenizer = HuggingFaceTokenizer("/path/to/tokenizer")
tokens = tokenizer.encode("Hello world", add_bos=True, add_eos=True)
text = tokenizer.decode(tokens)
vocab_size = tokenizer.vocab_size
```

### 2. HuggingFaceTextDataset (`torchtitan/hf_datasets/text_datasets.py`)

An `IterableDataset` that tokenizes during iteration:

```python
class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __iter__(self):
        for sample in self._data:
            # 1. Process sample text using dataset-specific processor
            sample_text = self._text_processor(sample)

            # 2. Tokenize with BOS/EOS
            sample_tokens = self._tokenizer.encode(
                sample_text, add_bos=True, add_eos=True
            )

            # 3. Add to buffer for sequence packing
            self._token_buffer.extend(sample_tokens)

            # 4. Yield complete sequences
            while len(self._token_buffer) >= self.seq_len + 1:
                x = torch.LongTensor(self._token_buffer[:self.seq_len + 1])
                self._token_buffer = self._token_buffer[self.seq_len + 1:]

                # Split into input and label for causal LM
                yield {"input": x[:-1]}, x[1:]
```

### 3. Sequence Packing

TorchTitan uses **sequence packing** to maximize GPU utilization. Multiple documents are concatenated into fixed-length sequences:

```
Document 1: "The cat sat."     → [BOS, The, cat, sat, ., EOS]
Document 2: "A dog ran fast."  → [BOS, A, dog, ran, fast, ., EOS]

Token Buffer (concatenated):
[BOS, The, cat, sat, ., EOS, BOS, A, dog, ran, fast, ., EOS, ...]

With seq_len=8, yields:
┌─────────────────────────────────────────────────────────────┐
│ Sequence 1 (9 tokens from buffer):                         │
│   tokens: [BOS, The, cat, sat, ., EOS, BOS, A, dog]        │
│   input:  [BOS, The, cat, sat, ., EOS, BOS, A]     (0:8)   │
│   label:  [The, cat, sat, ., EOS, BOS, A, dog]     (1:9)   │
└─────────────────────────────────────────────────────────────┘
```

**Why seq_len + 1?** The dataset yields `seq_len + 1` tokens so that:
- `input = tokens[:-1]` (first `seq_len` tokens)
- `label = tokens[1:]` (last `seq_len` tokens, shifted by 1)

This creates the standard causal language modeling setup where each position predicts the next token.

### 4. Checkpointing Support

The dataset implements `Stateful` for checkpointable data loading:

```python
# Saved state includes:
{
    "token_buffer": [...],  # Remaining tokens in buffer
    "sample_idx": 12345,    # For map-style datasets
    "data": {...}           # For iterable datasets (HF state_dict)
}
```

This enables resuming training from the exact position in the dataset.

---

## Adding Your Dataset

You'll need to add three components:
1. A dataset loader function
2. A sample processor function
3. A dataset configuration entry

### 1. Define Dataset Loader

Create a function that specifies how to load your dataset:

```python
def load_wikipedia_dataset(dataset_path: str):
    """Load Wikipedia dataset with specific configuration."""
    logger.info("Loading Wikipedia dataset...")
    return load_dataset(
        dataset_path,
        name="20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
```

### 2. Define Sample Processor

Create a function that extracts text from individual samples:

```python
def process_wikipedia_text(sample: dict[str, Any]) -> str:
    """Process Wikipedia dataset sample text."""
    return f"{sample['title']}\n\n{sample['text']}"
```

### 3. Register Your Dataset

Add your dataset configuration to the `DATASETS` dictionary in `torchtitan/hf_datasets/text_datasets.py`:

```python
DATASETS = {
    # ... existing datasets ...
    "wikipedia": DatasetConfig(
        path="wikipedia",  # default HuggingFace dataset path
        loader=load_wikipedia_dataset,
        sample_processor=process_wikipedia_text,
    ),
}
```

### 4. Configure Your Training

In your training configuration file (`.toml`), set your dataset:

```toml
[training]
dataset = "wikipedia"
# dataset_path = "custom/path"  # Optional: override default path
```

---

## Built-in Datasets

| Dataset | Name | Description |
|---------|------|-------------|
| C4 (train) | `c4` | AllenAI C4 dataset, English, streaming |
| C4 (validation) | `c4_validation` | C4 validation split for evaluation |
| C4 (test) | `c4_test` | Local test dataset for CI |

---

## Key Points

- **On-the-fly tokenization**: Text is tokenized during iteration, not upfront
- **Sequence packing**: Multiple documents are concatenated for efficiency
- **Streaming support**: Use `streaming=True` for large datasets to manage memory
- **Checkpointable**: Dataset state is saved/restored with training checkpoints
- **Flexible processors**: The `sample_processor` function lets you combine multiple fields

### DatasetConfig Fields

```python
@dataclass
class DatasetConfig:
    path: str           # Default HuggingFace dataset path
    loader: Callable    # Function to load the dataset
    sample_processor: Callable  # Function to extract text from samples
```

---

## Configuration Options

Relevant config options in your TOML file:

```toml
[model]
hf_assets_path = "./assets/hf/Llama3.1-8B"  # Tokenizer location

[training]
dataset = "c4"              # Dataset name (must be in DATASETS dict)
dataset_path = ""           # Optional: override default path
seq_len = 8192              # Sequence length for packing
local_batch_size = 2        # Batch size per GPU

[training.dataloader]
num_workers = 2             # DataLoader workers
pin_memory = false          # Pin memory for faster GPU transfer
```

---

## Why On-the-Fly Tokenization?

TorchTitan tokenizes during iteration rather than pre-tokenizing because:

1. **Memory efficiency**: No need to store tokenized versions of large datasets
2. **Flexibility**: Can change `seq_len` without re-processing the entire dataset
3. **Streaming compatibility**: Works with streaming datasets that don't fit in memory
4. **Checkpointing**: Can resume from any point by saving buffer state

The trade-off is slightly higher CPU usage during training, but this is typically not a bottleneck when training on GPUs.
