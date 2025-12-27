"""
GPT-2 Tiktoken Tokenizer
========================

This module provides a tiktoken-based tokenizer for GPT-2 that can be used as an
alternative to the HuggingFace tokenizer. Tiktoken is OpenAI's fast BPE tokenizer
library, which is 3-6x faster than comparable tokenizers.

GPT-2 Vocabulary:
- 256 raw byte tokens
- 50,000 BPE merge tokens
- 1 special token (<|endoftext|>) at position 50256
- Total: 50,257 tokens

Usage:
    # In your config TOML, set:
    # tokenizer = "tiktoken"

    # Or use directly:
    from torchtitan.experiments.gpt2.tokenizer import TiktokenTokenizer
    tokenizer = TiktokenTokenizer()
    tokens = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(tokens)
"""

from typing import Union

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


class TiktokenTokenizer(BaseTokenizer):
    """
    A tokenizer wrapper for tiktoken's GPT-2 encoding.

    This provides a fast, lightweight tokenizer that matches the original
    GPT-2 tokenizer exactly. It's useful when you don't need the full
    HuggingFace tokenizer infrastructure.

    Attributes:
        eos_id: End-of-sequence token ID (50256 for GPT-2)
        bos_id: Beginning-of-sequence token ID (same as eos_id for GPT-2)
    """

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize the tiktoken tokenizer.

        Args:
            encoding_name: The tiktoken encoding to use. Default is "gpt2".
                          Other options include "cl100k_base" (GPT-4), etc.
        """
        super().__init__()

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. "
                "Install it with: pip install tiktoken"
            )

        self._encoding = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name

        # GPT-2 uses <|endoftext|> as both BOS and EOS token
        # It's at position 50256 (the last token in the vocabulary)
        self.eos_id = self._encoding.eot_token  # End of text token
        self.bos_id = self.eos_id  # GPT-2 uses same token for BOS

        logger.info(
            f"Initialized tiktoken tokenizer with encoding '{encoding_name}', "
            f"vocab_size={self.vocab_size}, eos_id={self.eos_id}"
        )

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        **kwargs,
    ) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            text: The text to encode
            add_bos: Whether to prepend BOS token (default: False)
            add_eos: Whether to append EOS token (default: False)
            **kwargs: Additional arguments (ignored, for API compatibility)

        Returns:
            List of token IDs
        """
        tokens = self._encoding.encode(text)

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: list[int], **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode
            **kwargs: Additional arguments (ignored, for API compatibility)

        Returns:
            Decoded text string
        """
        return self._encoding.decode(tokens)

    @property
    def vocab_size(self) -> int:
        """Get the tokenizer vocabulary size (50257 for GPT-2)."""
        return self._encoding.n_vocab

    def get_vocab_size(self) -> int:
        """Get the tokenizer vocabulary size (50257 for GPT-2)."""
        return self._encoding.n_vocab

    def token_to_id(self, token: str) -> int:
        """Convert a token string to its ID."""
        ids = self._encoding.encode(token)
        if len(ids) != 1:
            raise ValueError(
                f"Token '{token}' encodes to multiple IDs: {ids}. "
                "Use encode() for multi-token strings."
            )
        return ids[0]

    def id_to_token(self, token_id: int) -> str:
        """Convert a token ID to its string representation."""
        return self._encoding.decode([token_id])

    def encode_batch(self, texts: list[str], **kwargs) -> list[list[int]]:
        """
        Encode multiple texts in batch.

        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments passed to encode()

        Returns:
            List of token ID lists
        """
        return [self.encode(text, **kwargs) for text in texts]

    def decode_batch(self, token_lists: list[list[int]], **kwargs) -> list[str]:
        """
        Decode multiple token lists in batch.

        Args:
            token_lists: List of token ID lists to decode
            **kwargs: Additional arguments passed to decode()

        Returns:
            List of decoded strings
        """
        return [self.decode(tokens, **kwargs) for tokens in token_lists]


def build_tiktoken_tokenizer(
    job_config: JobConfig,
) -> TiktokenTokenizer:
    """
    Build a tiktoken-based GPT-2 tokenizer.

    This is an alternative to build_hf_tokenizer that uses tiktoken instead
    of HuggingFace tokenizers. It's faster and doesn't require downloading
    tokenizer files.

    Args:
        job_config: Job configuration (encoding can be specified via config)

    Returns:
        TiktokenTokenizer instance
    """
    # Default to GPT-2 encoding
    encoding_name = "gpt2"

    # Could extend to support other encodings via config in the future
    # e.g., job_config.model.tiktoken_encoding

    return TiktokenTokenizer(encoding_name=encoding_name)
