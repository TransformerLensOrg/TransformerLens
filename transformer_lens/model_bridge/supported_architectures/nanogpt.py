from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class NanogptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NanoGPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the NanoGPT architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.b_Q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_K": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_V": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (n_head d_head) -> n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_proj.weight",
            ),
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),  # Word token embeddings
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),  # Positional embeddings
            "blocks": BlockBridge(
                name="transformer.h",  # Base path for blocks
                submodules={
                    "ln1": NormalizationBridge(
                        name="ln_1", config=self.cfg
                    ),  # Pre-attention layer norm
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),  # Pre-MLP layer norm
                    "attn": AttentionBridge(name="attn", config=self.cfg),  # Full attention module
                    "mlp": MLPBridge(name="mlp"),  # Full MLP module
                },
            ),
            "ln_final": NormalizationBridge(
                name="transformer.ln_f", config=self.cfg
            ),  # Final layer norm
            "unembed": UnembeddingBridge(name="lm_head"),  # Language model head
        }

    def convert_weights(self, remote_module: Any) -> dict[str, torch.Tensor]:
        # Nanogpt models saved after torch.compile() have this unwanted prefix
        # This is a simple way to remove it
        unwanted_prefix = "_orig_mod."
        state_dict: dict[str, torch.Tensor] = (
            remote_module.state_dict() if hasattr(remote_module, "state_dict") else remote_module
        )
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        return super().convert_weights(remote_module)  # type: ignore[misc]
