"""Neo architecture adapter."""

from typing import Any

import einops
import torch

from transformer_lens.conversion_utils.conversion_steps import (
    BaseTensorConversion,
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class NeoLinearTransposeConversion(BaseTensorConversion):
    """Transpose Linear weights to Conv1D format and rearrange for GPT-Neo.

    GPT-Neo uses standard PyTorch Linear layers with weights shaped [out_features, in_features].
    This conversion transposes them to Conv1D format [in_features, out_features] and then
    applies einops rearrangement for attention heads.
    """

    def __init__(self, rearrange_pattern: str | None = None, **axes_lengths):
        """Initialize the conversion.

        Args:
            rearrange_pattern: Optional einops pattern for rearrangement after transpose
            **axes_lengths: Additional axes lengths for einops (e.g., n=n_heads)
        """
        super().__init__()
        self.rearrange_pattern = rearrange_pattern
        self.axes_lengths = axes_lengths

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Transpose from Linear to Conv1D format and optionally rearrange."""
        # Transpose: [out_features, in_features] -> [in_features, out_features]
        transposed = input_value.T

        # Apply rearrangement if specified
        if self.rearrange_pattern:
            return einops.rearrange(transposed, self.rearrange_pattern, **self.axes_lengths)

        return transposed

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Revert rearrangement and transpose back to Linear format."""
        result = input_value

        # Reverse rearrangement if specified
        if self.rearrange_pattern:
            # Reverse the einops pattern
            left, right = self.rearrange_pattern.split("->")
            reversed_pattern = f"{right.strip()} -> {left.strip()}"
            result = einops.rearrange(result, reversed_pattern, **self.axes_lengths)

        # Transpose back: [in_features, out_features] -> [out_features, in_features]
        return result.T


class NeoArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Neo models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Neo architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # GPT-Neo uses BOS tokens (inherits default_prepend_bos = True)

        self.weight_processing_conversions = {
            # Property access keys (used by component tree) - for attention
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=NeoLinearTransposeConversion(
                    "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                ),
                source_key="transformer.h.{i}.attn.attention.q_proj.weight",
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=NeoLinearTransposeConversion(
                    "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                ),
                source_key="transformer.h.{i}.attn.attention.k_proj.weight",
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=NeoLinearTransposeConversion(
                    "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                ),
                source_key="transformer.h.{i}.attn.attention.v_proj.weight",
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=NeoLinearTransposeConversion(
                    "(n h) d_model -> n h d_model", n=self.cfg.n_heads
                ),
                source_key="transformer.h.{i}.attn.attention.out_proj.weight",
            ),
            # Property access keys - for MLP
            "blocks.{i}.mlp.W_in": ParamProcessingConversion(
                tensor_conversion=NeoLinearTransposeConversion(),  # Just transpose, no rearrange needed,
                source_key="transformer.h.{i}.mlp.c_fc.weight",
            ),
            "blocks.{i}.mlp.W_out": ParamProcessingConversion(
                tensor_conversion=NeoLinearTransposeConversion(),  # Just transpose, no rearrange needed,
                source_key="transformer.h.{i}.mlp.c_proj.weight",
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.attention.q_proj.bias",
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.attention.k_proj.bias",
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
                source_key="transformer.h.{i}.attn.attention.v_proj.bias",
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                config=self.cfg,
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "attn": AttentionBridge(
                        name="attn.attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="c_fc"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
