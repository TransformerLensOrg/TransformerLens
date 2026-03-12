"""GPT2 architecture adapter."""

from typing import Any

import einops
import torch

from transformer_lens.conversion_utils.conversion_steps import (
    BaseTensorConversion,
    RearrangeTensorConversion,
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class QKVSplitRearrangeConversion(BaseTensorConversion):
    """Custom conversion that splits QKV tensor and then rearranges.

    Handles two input formats:
    - Combined QKV tensor (from HuggingFace): one dimension is ~3x the other.
      Splits into Q/K/V parts, then rearranges to TL format.
    - Already-split tensor (from bridge state dict): nn.Linear format
      [n_heads*d_head, d_model]. Rearranges directly to TL format.
    """

    def __init__(self, qkv_index: int, rearrange_pattern: str, **axes_lengths):
        """Initialize the conversion.

        Args:
            qkv_index: Index of Q (0), K (1), or V (2) in the QKV tensor
            rearrange_pattern: Einops pattern for rearrangement (Conv1D format)
            **axes_lengths: Additional axes lengths for einops
        """
        super().__init__()
        self.qkv_index = qkv_index
        self.rearrange_pattern = rearrange_pattern
        self.axes_lengths = axes_lengths

    def _is_combined_qkv(self, tensor: torch.Tensor) -> bool:
        """Check if a tensor is a combined QKV tensor vs already-split."""
        if tensor.ndim == 2:
            d0, d1 = tensor.shape
            return d1 > d0 * 2 or d0 > d1 * 2
        if tensor.ndim == 1:
            n = self.axes_lengths.get("n", 1)
            # Combined bias has 3x the expected individual size
            return tensor.shape[0] % 3 == 0 and tensor.shape[0] > n * 3
        return False

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Split QKV tensor and rearrange the selected part."""
        if not self._is_combined_qkv(input_value):
            # Already-split nn.Linear format — transpose rearrange pattern:
            return einops.rearrange(
                input_value, "(n h) d_model -> n d_model h", **self.axes_lengths
            )

        # Combined QKV tensor — split then rearrange
        if len(input_value.shape) == 2:
            # Weight tensor: [d_model, 3*d_model] -> split along dim=1
            split_dim = 1 if input_value.shape[1] > input_value.shape[0] else 0
        elif len(input_value.shape) == 1:
            # Bias tensor: [3*n_heads*d_head] -> split along dim=0
            split_dim = 0
        else:
            raise ValueError(f"Unexpected tensor shape: {input_value.shape}")

        qkv_parts = torch.tensor_split(input_value, 3, dim=split_dim)
        selected_part = qkv_parts[self.qkv_index]
        return einops.rearrange(selected_part, self.rearrange_pattern, **self.axes_lengths)

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Revert from TL format [n_heads, d_model, d_head] to nn.Linear format."""
        if input_value.ndim == 3:
            return einops.rearrange(
                input_value, "n d_model h -> (n h) d_model", **self.axes_lengths
            )
        if input_value.ndim == 2:
            # Bias in TL format [n_heads, d_head] -> [n_heads*d_head]
            return einops.rearrange(input_value, "n h -> (n h)", **self.axes_lengths)
        return input_value


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT2 models.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    GPT-2 models HAVE biases on ALL linear layers:

    ✓ blocks.{i}.attn.b_Q - Has bias (from combined c_attn.bias)
    ✓ blocks.{i}.attn.b_K - Has bias (from combined c_attn.bias)
    ✓ blocks.{i}.attn.b_V - Has bias (from combined c_attn.bias)
    ✓ blocks.{i}.attn.b_O - Has bias (c_proj.bias)
    ✓ blocks.{i}.mlp.b_in - Has bias (c_fc.bias)
    ✓ blocks.{i}.mlp.b_out - Has bias (c_proj.bias)
    ✓ blocks.{i}.ln1.b - LayerNorm has bias
    ✓ blocks.{i}.ln2.b - LayerNorm has bias
    ✓ ln_final.b - LayerNorm has bias

    No optional parameters - all biases exist in GPT-2.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT2 architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # GPT-2 uses BOS tokens (inherits default_prepend_bos = True)

        # Set default config for GPT2 models
        self.default_cfg = {
            "uses_split_attention": True,  # GPT-2 uses combined QKV attention that needs splitting
        }

        # GPT-2 uses combined QKV weights in HuggingFace format
        self.uses_combined_qkv = True

        # Set config variable to indicate that attention weights are split (use TransformerLens format processing)
        self.cfg.split_attention_weights = True

        from transformer_lens.conversion_utils.param_processing_conversion import (
            ParamProcessingConversion,
        )

        self.weight_processing_conversions = {
            # Q/K/V weights - split from joint qkv.weight and rearrange
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=QKVSplitRearrangeConversion(
                    qkv_index=0,
                    rearrange_pattern="d_model (n h) -> n d_model h",
                    n=self.cfg.n_heads,
                ),
                source_key="blocks.{i}.attn.qkv.weight",
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=QKVSplitRearrangeConversion(
                    qkv_index=1,
                    rearrange_pattern="d_model (n h) -> n d_model h",
                    n=self.cfg.n_heads,
                ),
                source_key="blocks.{i}.attn.qkv.weight",
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=QKVSplitRearrangeConversion(
                    qkv_index=2,
                    rearrange_pattern="d_model (n h) -> n d_model h",
                    n=self.cfg.n_heads,
                ),
                source_key="blocks.{i}.attn.qkv.weight",
            ),
            # Q/K/V biases - split from joint qkv.bias and reshape
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    pattern="(index head) -> index head",
                    index=self.cfg.n_heads,
                    head=self.cfg.d_head,
                ),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    pattern="(index head) -> index head",
                    index=self.cfg.n_heads,
                    head=self.cfg.d_head,
                ),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    pattern="(index head) -> index head",
                    index=self.cfg.n_heads,
                    head=self.cfg.d_head,
                ),
            ),
            # O weight - rearrange from 2D to 3D
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    pattern="(n h) m -> n h m", n=self.cfg.n_heads
                ),
            ),
            # Unembed weight - transpose from [d_model, d_vocab] to [d_vocab, d_model]
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
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
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "qkv": LinearBridge(name="c_attn"),
                            "o": LinearBridge(name="c_proj"),
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
