"""GPTBigCode architecture adapter."""

from typing import Any

import einops
import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
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


class MQAQKVConversionRule(BaseTensorConversion):
    """Rearranges Q/K/V activations for MQA.

    Q output has embed_dim features -> rearrange with n=n_heads.
    K/V output has head_dim features (1 KV head) -> rearrange with n=1.
    """

    def __init__(self, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

    def handle_conversion(self, input_value: torch.Tensor, *_: Any) -> torch.Tensor:
        if input_value.ndim == 4:
            return input_value  # already [batch, seq, heads, head_dim]
        if input_value.ndim != 3:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {input_value.ndim}D with shape {input_value.shape}"
            )
        last_dim: int = input_value.shape[2]
        # Q: last_dim == n_heads * d_head; K/V: last_dim == d_head (1 head)
        n = self.n_heads if last_dim == self.n_heads * self.d_head else 1
        return einops.rearrange(input_value, "batch seq (n h) -> batch seq n h", n=n)

    def revert(self, input_value: torch.Tensor, *_: Any) -> torch.Tensor:
        if input_value.ndim == 3:
            return input_value
        return einops.rearrange(input_value, "batch seq n h -> batch seq (n h)")


class GPTBigCodeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPTBigCode models.

    GPTBigCode is a GPT-2 variant using Multi-Query Attention (MQA): a single
    fused c_attn projection whose output splits asymmetrically into
    [embed_dim, head_dim, head_dim] for Q/K/V (rather than three equal thirds).
    All other structure (module paths, LayerNorm, learned pos embeddings,
    standard MLP) is identical to GPT-2.

    All public models use multi_query=True (1 KV head). The adapter assumes
    MQA throughout.

    All linear layers have biases (c_attn, c_proj, c_fc, mlp.c_proj).
    lm_head has no bias and its weight is tied to transformer.wte.weight.

    Weight layout difference from GPT-2: GPTBigCode uses nn.Linear (weights
    stored [out, in]) rather than GPT-2's Conv1D ([in, out]), so no unembed
    weight transpose is needed.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        self.cfg.eps_attr = "layer_norm_epsilon"
        self.cfg.n_key_value_heads = 1  # MQA: always 1 KV head

        # Mirror GPT-2 combined-QKV flags
        self.default_cfg = {"uses_split_attention": True}
        self.uses_combined_qkv = True
        self.cfg.split_attention_weights = True

        # Use the base helper; n_kv_heads=1 gives correct (n h) m -> n m h with n=1 for K/V
        self.weight_processing_conversions: dict[str, ParamProcessingConversion] = {  # type: ignore[assignment]
            **self._qkvo_weight_conversions(n_kv_heads=1),
        }

        _mqa_rule = MQAQKVConversionRule(n_heads=self.cfg.n_heads, d_head=self.cfg.d_head)

        # GPTBigCode's HF eager_attention_forward only applies causal masking
        # when attention_mask is not None. Setting requires_attention_mask with
        # attention_mask_4d ensures component tests provide a 4D mask so both
        # HF and bridge forward passes receive compatible mask shapes.
        _attn_bridge = JointQKVAttentionBridge(
            name="attn",
            config=self.cfg,
            split_qkv_matrix=self._split_qkv_matrix,
            qkv_conversion_rule=_mqa_rule,
            requires_attention_mask=True,
            submodules={
                "qkv": LinearBridge(name="c_attn"),
                "o": LinearBridge(name="c_proj"),
            },
        )
        _attn_bridge.attention_mask_4d = True

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                config=self.cfg,
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "attn": _attn_bridge,
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

    def _split_qkv_matrix(
        self, original_attention_component: Any
    ) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Split MQA c_attn into separate Q, K, V linears.

        c_attn is nn.Linear with weight shape [embed_dim + 2*head_dim, embed_dim].
        Split along dim=0 (output features): [embed_dim, head_dim, head_dim].

        Returns nn.Linear modules with shapes:
          Q: [embed_dim, embed_dim]  (n_heads * d_head output features)
          K: [head_dim, embed_dim]   (1 KV head)
          V: [head_dim, embed_dim]   (1 KV head)
        """
        # Guard against multi_query=False checkpoints (MHA), which would require
        # an equal 3-way split and different hook shapes.
        assert getattr(original_attention_component, "multi_query", True), (
            "GPTBigCodeArchitectureAdapter only supports multi_query=True models. "
            "For multi_query=False checkpoints, a separate MHA adapter is needed."
        )

        c_attn = original_attention_component.c_attn
        embed_dim = self.cfg.d_model
        head_dim = self.cfg.d_head

        q_w, k_w, v_w = c_attn.weight.split([embed_dim, head_dim, head_dim], dim=0)

        has_bias = c_attn.bias is not None
        q_b: torch.Tensor | None = None
        k_b: torch.Tensor | None = None
        v_b: torch.Tensor | None = None
        if has_bias:
            q_b, k_b, v_b = c_attn.bias.split([embed_dim, head_dim, head_dim])

        def _make_linear(w: torch.Tensor, b: torch.Tensor | None) -> nn.Linear:
            lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
            lin.weight = nn.Parameter(w)
            if b is not None:
                lin.bias = nn.Parameter(b)
            return lin

        return _make_linear(q_w, q_b), _make_linear(k_w, k_b), _make_linear(v_w, v_b)
