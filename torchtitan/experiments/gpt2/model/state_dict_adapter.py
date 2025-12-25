"""
GPT-2 State Dict Adapter
========================

Converts between TorchTitan GPT-2 state dict format and HuggingFace GPT-2 format.

TorchTitan GPT-2 Keys:
- tok_embeddings.weight
- pos_embeddings.weight
- layers.{i}.ln_1.{weight,bias}
- layers.{i}.attn.c_attn.{weight,bias}
- layers.{i}.attn.c_proj.{weight,bias}
- layers.{i}.ln_2.{weight,bias}
- layers.{i}.mlp.c_fc.{weight,bias}
- layers.{i}.mlp.c_proj.{weight,bias}
- norm.{weight,bias}
- output.weight

HuggingFace GPT-2 Keys:
- wte.weight (token embeddings)
- wpe.weight (position embeddings)
- h.{i}.ln_1.{weight,bias}
- h.{i}.attn.c_attn.{weight,bias}
- h.{i}.attn.c_proj.{weight,bias}
- h.{i}.ln_2.{weight,bias}
- h.{i}.mlp.c_fc.{weight,bias}
- h.{i}.mlp.c_proj.{weight,bias}
- ln_f.{weight,bias}
- lm_head.weight (tied with wte.weight in original GPT-2)
"""

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import GPT2ModelArgs


class GPT2StateDictAdapter(StateDictAdapter):
    """State dict adapter for converting between TorchTitan and HuggingFace GPT-2 formats."""

    def __init__(
        self,
        model_args: GPT2ModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan GPT-2 state dict to HuggingFace format.

        Args:
            state_dict: TorchTitan state dict with 'model.' prefix

        Returns:
            HuggingFace format state dict
        """
        hf_state_dict = {}

        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            if key.startswith("model."):
                key = key[6:]

            # Skip output.weight when weight tying is enabled - it's the same
            # tensor as tok_embeddings.weight, so saving it would be redundant
            if self.model_args.weight_tying and key == "output.weight":
                continue

            # Convert key names
            new_key = self._tt_to_hf_key(key)
            if new_key is not None:
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace GPT-2 state dict to TorchTitan format.

        Args:
            hf_state_dict: HuggingFace format state dict

        Returns:
            TorchTitan state dict with 'model.' prefix
        """
        # Handle weight tying: if lm_head.weight is missing but weight_tying
        # is enabled, use wte.weight for both
        if (
            self.model_args.weight_tying
            and "lm_head.weight" not in hf_state_dict
            and "wte.weight" in hf_state_dict
        ):
            hf_state_dict = dict(hf_state_dict)  # Make a copy to avoid modifying original
            hf_state_dict["lm_head.weight"] = hf_state_dict["wte.weight"]

        state_dict = {}

        for key, value in hf_state_dict.items():
            new_key = self._hf_to_tt_key(key)
            if new_key is not None:
                state_dict[f"model.{new_key}"] = value

        return state_dict

    def _tt_to_hf_key(self, key: str) -> str | None:
        """Convert a single TorchTitan key to HuggingFace format."""

        # Token embeddings
        if key == "tok_embeddings.weight":
            return "wte.weight"

        # Position embeddings
        if key == "pos_embeddings.weight":
            return "wpe.weight"

        # Final layer norm
        if key == "norm.weight":
            return "ln_f.weight"
        if key == "norm.bias":
            return "ln_f.bias"

        # Output projection (lm_head)
        # Note: In HF GPT-2, lm_head.weight is tied with wte.weight
        # We still export it separately for compatibility
        if key == "output.weight":
            return "lm_head.weight"

        # Transformer layers: layers.{i}.* -> h.{i}.*
        layer_match = re.match(r"layers\.(\d+)\.(.*)", key)
        if layer_match:
            layer_idx = layer_match.group(1)
            rest = layer_match.group(2)
            return f"h.{layer_idx}.{rest}"

        # Unknown key - skip
        return None

    def _hf_to_tt_key(self, key: str) -> str | None:
        """Convert a single HuggingFace key to TorchTitan format."""

        # Token embeddings
        if key == "wte.weight":
            return "tok_embeddings.weight"

        # Position embeddings
        if key == "wpe.weight":
            return "pos_embeddings.weight"

        # Final layer norm
        if key == "ln_f.weight":
            return "norm.weight"
        if key == "ln_f.bias":
            return "norm.bias"

        # Output projection (lm_head)
        if key == "lm_head.weight":
            return "output.weight"

        # Transformer layers: h.{i}.* -> layers.{i}.*
        layer_match = re.match(r"h\.(\d+)\.(.*)", key)
        if layer_match:
            layer_idx = layer_match.group(1)
            rest = layer_match.group(2)
            return f"layers.{layer_idx}.{rest}"

        # Unknown key - skip
        return None
