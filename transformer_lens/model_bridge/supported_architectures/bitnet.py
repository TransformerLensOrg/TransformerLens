"""BitNet b1.58 (``BitNetForCausalLM``) adapter: llama layout plus attn/ffn
sub-layer RMSNorms (attn_sub_norm reapplied by an adapter-local attention bridge)."""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)
from transformer_lens.utilities.quantization import unreadable_weight_reason


class _BitNetAttentionBridge(PositionEmbeddingsAttentionBridge):
    """Applies BitNet's attn_sub_norm before the output projection.

    The generic reconstruction goes straight from attention output to o_proj;
    BitNet inserts an RMSNorm in between.
    """

    def _pre_output_projection(self, attn_output: torch.Tensor) -> torch.Tensor:
        oc = self.original_component
        sub_norm = getattr(oc, "attn_sub_norm", None) if oc is not None else None
        if isinstance(sub_norm, torch.nn.Module):
            attn_output = sub_norm(attn_output)
        return attn_output


class BitNetArchitectureAdapter(LlamaArchitectureAdapter):
    """Architecture adapter for BitNetForCausalLM models."""

    _attention_bridge_cls = _BitNetAttentionBridge
    _testing_eager = "config"

    # Sub-layer norms are incompatible with HT-style processed-weight
    # attention, so compatibility-mode equivalence (Phase 3) is out of scope.
    applicable_phases: list[int] = [1, 2, 4]

    def __init__(self, cfg: Any) -> None:
        """Initialize the BitNet architecture adapter."""
        super().__init__(cfg)

        # Sub-layer norms sit between activations and output projections;
        # standard LN folding and W_O centering do not model them.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False

    def prepare_model(self, hf_model: Any) -> None:
        """Refuse packed 1.58-bit checkpoints, which need BitNet dequant kernels.

        The flagship microsoft/bitnet-b1.58-2B-4T stores `weight` as packed
        uint8 with a collapsed first dim (out_features // 4) plus a separate
        weight_scale, so every weight-space read reshapes it into a
        wrong-but-plausible matrix rather than failing. The registry records
        this checkpoint at 0% on the forward-pass phase for exactly that reason.
        """
        super().prepare_model(hf_model)
        for name, module in hf_model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            if unreadable_weight_reason(weight) is not None:
                raise NotImplementedError(
                    f"BitNet checkpoint stores packed weights ({name}); "
                    "TransformerLens needs the dequantized sibling — use "
                    "microsoft/bitnet-b1.58-2B-4T-bf16."
                )
            break
