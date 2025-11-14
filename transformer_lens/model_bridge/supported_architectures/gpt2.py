"""GPT2 architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    BaseHookConversion,
    HookConversionSet,
    RearrangeHookConversion,
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
                # Original parameter names (for compatibility)
                "pos_embed.pos": "transformer.wpe.weight",
                "embed.e": "transformer.wte.weight",
                "blocks.{i}.ln1.weight": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.bias": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.q.weight": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "(n h) m-> n m h",
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.k.weight": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "(n h) m-> n m h",
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.v.weight": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "(n h) m-> n m h",
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.o.weight": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeHookConversion("(n h) m -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.q.bias": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(n d_head) -> n d_head", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k.bias": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(n d_head) -> n d_head", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v.bias": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(n d_head) -> n d_head", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o.bias": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln2.weight": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.bias": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.input.weight": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.input.bias": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.weight": "transformer.ln_f.weight",
                "ln_final.bias": "transformer.ln_f.bias",
                "unembed.weight": (
                    "lm_head.weight",
                    RearrangeHookConversion("d_model d_vocab -> d_vocab d_model"),
                ),
                "unembed.bias": "lm_head.bias",
                # TransformerLens parameter names (for weight processing functions)
                "embed.W_E": "transformer.wte.weight",
                "pos_embed.W_pos": "transformer.wpe.weight",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        qkv_index=0,  # Q is the first part
                        rearrange_pattern="m (i h) -> i m h",
                        i=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        qkv_index=1,  # K is the second part
                        rearrange_pattern="m (i h) -> i m h",
                        i=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    QKVSplitRearrangeConversion(
                        qkv_index=2,  # V is the third part
                        rearrange_pattern="m (i h) -> i m h",
                        i=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeHookConversion("(i h) m -> i h m", i=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(
                        qkv_index=0,  # Q bias is the first part
                        n_heads=self.cfg.n_heads,
                        d_head=self.cfg.d_head,
                    ),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(
                        qkv_index=1,  # K bias is the second part
                        n_heads=self.cfg.n_heads,
                        d_head=self.cfg.d_head,
                    ),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    QKVBiasConversion(
                        qkv_index=2,  # V bias is the third part
                        n_heads=self.cfg.n_heads,
                        d_head=self.cfg.d_head,
                    ),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": (
                    "lm_head.weight",
                    RearrangeHookConversion("d_model d_vocab -> d_vocab d_model"),
                ),
                "unembed.b_U": "lm_head.bias",
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
