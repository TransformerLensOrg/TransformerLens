"""GPT2 architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    BaseHookConversion,
    HookConversionSet,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    Conv1DBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class QKVSplitRearrangeConversion(BaseHookConversion):
    """Custom conversion that splits QKV tensor and then rearranges."""

    def __init__(self, qkv_index: int, rearrange_pattern: str, **axes_lengths):
        """Initialize the conversion.

        Args:
            qkv_index: Index of Q (0), K (1), or V (2) in the QKV tensor
            rearrange_pattern: Einops pattern for rearrangement
            **axes_lengths: Additional axes lengths for einops
        """
        super().__init__()
        self.qkv_index = qkv_index
        self.rearrange_pattern = rearrange_pattern
        self.axes_lengths = axes_lengths

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Split QKV tensor and rearrange the selected part."""
        # Determine the split dimension based on tensor shape
        if len(input_value.shape) == 2:
            # Weight tensor: [d_model, 3*d_model] -> split along dim=1
            split_dim = 1
        elif len(input_value.shape) == 1:
            # Bias tensor: [3*n_heads*d_head] -> split along dim=0
            split_dim = 0
        else:
            raise ValueError(f"Unexpected tensor shape: {input_value.shape}")

        # Split the QKV tensor
        qkv_parts = torch.chunk(input_value, 3, dim=split_dim)
        selected_part = qkv_parts[self.qkv_index]

        # Apply rearrangement
        import einops

        return einops.rearrange(selected_part, self.rearrange_pattern, **self.axes_lengths)


class QKVBiasConversion(BaseHookConversion):
    """Custom conversion for QKV biases that matches the original GPT-2 logic."""

    def __init__(self, qkv_index: int, n_heads: int, d_head: int):
        """Initialize the conversion.

        Args:
            qkv_index: Index of Q (0), K (1), or V (2) in the QKV tensor
            n_heads: Number of attention heads
            d_head: Dimension of each head
        """
        super().__init__()
        self.qkv_index = qkv_index
        self.n_heads = n_heads
        self.d_head = d_head

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Convert QKV bias following the original GPT-2 logic."""
        import einops

        # Original logic: rearrange the entire bias tensor first, then split by QKV
        qkv_bias = einops.rearrange(
            input_value,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.n_heads,
            head=self.d_head,
        )
        # Return the selected QKV part
        return qkv_bias[self.qkv_index]


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

        self.conversion_rules = HookConversionSet(
            {
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        0, "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        1, "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        2, "d_model (n h) -> n d_model h", n=self.cfg.n_heads
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(0, self.cfg.n_heads, self.cfg.d_head),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(1, self.cfg.n_heads, self.cfg.d_head),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(2, self.cfg.n_heads, self.cfg.d_head),
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
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        split_qkv_matrix=self.split_qkv_matrix,
                        submodules={
                            "qkv": Conv1DBridge(name="c_attn"),
                            "o": Conv1DBridge(name="c_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": Conv1DBridge(name="c_fc"),
                            "out": Conv1DBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def split_qkv_matrix(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        """Split the QKV matrix into separate linear transformations.

        Args:
            attention_component: The original attention layer component
        Returns:
            Tuple of nn.Linear modules for Q, K, and V transformations (output 3D tensors)
        """

        # Keep mypy happy
        assert original_attention_component is not None
        assert original_attention_component.c_attn is not None

        qkv_weights = original_attention_component.c_attn.weight

        # Keep mypy happy
        assert isinstance(qkv_weights, torch.Tensor)

        # Original qkv_weights shape: [d_model, 3 * d_model]
        # Split into three equal parts along dimension 1 to get Q, K, V weights
        W_Q, W_K, W_V = torch.tensor_split(qkv_weights, 3, dim=1)

        qkv_bias = original_attention_component.c_attn.bias

        # Keep mypy happy
        assert isinstance(qkv_bias, torch.Tensor)

        # Original qkv_bias shape: [3 * n_head * d_head]
        # Reshape to [3, n_head * d_head] to split by Q, K, V
        qkv_bias = qkv_bias.reshape(3, self.cfg.n_heads * self.cfg.d_head)
        b_Q, b_K, b_V = qkv_bias[0, :], qkv_bias[1, :], qkv_bias[2, :]

        # Create plain nn.Linear modules that output 3D tensors [batch, seq, d_model]
        W_Q_transformation = torch.nn.Linear(W_Q.shape[0], W_Q.shape[1], bias=True)
        W_Q_transformation.weight = torch.nn.Parameter(W_Q.T)
        W_Q_transformation.bias = torch.nn.Parameter(b_Q)

        W_K_transformation = torch.nn.Linear(W_K.shape[0], W_K.shape[1], bias=True)
        W_K_transformation.weight = torch.nn.Parameter(W_K.T)
        W_K_transformation.bias = torch.nn.Parameter(b_K)

        W_V_transformation = torch.nn.Linear(W_V.shape[0], W_V.shape[1], bias=True)
        W_V_transformation.weight = torch.nn.Parameter(W_V.T)
        W_V_transformation.bias = torch.nn.Parameter(b_V)

        return W_Q_transformation, W_K_transformation, W_V_transformation
