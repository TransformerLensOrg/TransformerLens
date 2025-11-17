"""Neo architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    BaseHookConversion,
    HookConversionSet,
    RearrangeHookConversion,
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


class NeoLinearTransposeConversion(BaseHookConversion):
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
            import einops

            return einops.rearrange(transposed, self.rearrange_pattern, **self.axes_lengths)

        return transposed

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Revert rearrangement and transpose back to Linear format."""
        result = input_value

        # Reverse rearrangement if specified
        if self.rearrange_pattern:
            import einops

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

        self.conversion_rules = HookConversionSet(
            {
                # Property access keys (used by component tree) - for attention
                "blocks.{i}.attn.q.weight": (
                    "transformer.h.{i}.attn.attention.q_proj.weight",
                    NeoLinearTransposeConversion(
                        "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.k.weight": (
                    "transformer.h.{i}.attn.attention.k_proj.weight",
                    NeoLinearTransposeConversion(
                        "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.v.weight": (
                    "transformer.h.{i}.attn.attention.v_proj.weight",
                    NeoLinearTransposeConversion(
                        "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.o.weight": (
                    "transformer.h.{i}.attn.attention.out_proj.weight",
                    NeoLinearTransposeConversion(
                        "(n h) d_model -> n h d_model", n=self.cfg.n_heads
                    ),
                ),
                # Property access keys - for MLP
                "blocks.{i}.mlp.in.weight": (
                    "transformer.h.{i}.mlp.c_fc.weight",
                    NeoLinearTransposeConversion(),  # Just transpose, no rearrange needed
                ),
                "blocks.{i}.mlp.out.weight": (
                    "transformer.h.{i}.mlp.c_proj.weight",
                    NeoLinearTransposeConversion(),  # Just transpose, no rearrange needed
                ),
                # Weight processing keys (W_Q, W_K, W_V, W_O style) - for weight processing
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.attention.q_proj.weight",
                    NeoLinearTransposeConversion(
                        "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.attention.k_proj.weight",
                    NeoLinearTransposeConversion(
                        "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.attention.v_proj.weight",
                    NeoLinearTransposeConversion(
                        "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.attention.out_proj.weight",
                    NeoLinearTransposeConversion(
                        "(n h) d_model -> n h d_model", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.attention.q_proj.bias",
                    RearrangeHookConversion("(n h) -> n h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.attention.k_proj.bias",
                    RearrangeHookConversion("(n h) -> n h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.attention.v_proj.bias",
                    RearrangeHookConversion("(n h) -> n h", n=self.cfg.n_heads),
                ),
                # MLP weight processing keys
                "blocks.{i}.mlp.W_in": (
                    "transformer.h.{i}.mlp.c_fc.weight",
                    NeoLinearTransposeConversion(),  # Just transpose, no rearrange needed
                ),
                "blocks.{i}.mlp.W_out": (
                    "transformer.h.{i}.mlp.c_proj.weight",
                    NeoLinearTransposeConversion(),  # Just transpose, no rearrange needed
                ),
            }
        )

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
