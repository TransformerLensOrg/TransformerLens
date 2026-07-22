"""StableLM architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class StableLmArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for StableLM models.

    StableLM uses a Llama-like architecture with separate Q/K/V projections and
    gated MLP, but differs in using standard LayerNorm (not RMSNorm) and partial
    rotary embeddings (25% of head dimensions by default).

    Supports optional features:
    - Grouped Query Attention (num_key_value_heads != num_attention_heads)
    - QKV bias (use_qkv_bias=True on some models like stable-code-3b)
    - Parallel residual connections (use_parallel_residual=True)
    - Per-head QK LayerNorm (qk_layernorm=True)

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_K - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_V - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP up_proj
    - blocks.{i}.mlp.b_gate - No bias on MLP gate_proj
    - blocks.{i}.mlp.b_out - No bias on MLP down_proj
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the StableLM architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        # The bridge reimplements attention; the HF reference must run the
        # matching eager math.
        self.cfg.attn_implementation = "eager"

        n_kv_heads = getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
            # Bias conversions for models with use_qkv_bias=True
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv_heads),
            ),
        }

        # When parallel_attn_mlp=True (HF: use_parallel_residual=True), both attn
        # and MLP read from ln1 output:
        #   x = x + attn(ln1(x)) + mlp(ln1(x))
        # When False, they are sequential with separate norms:
        #   x = x + attn(ln1(x)); x = x + mlp(ln2(x))
        # HF sets post_attention_layernorm=None when use_parallel_residual=True,
        # so we must not include ln2 in that case.
        use_parallel_residual = getattr(cfg, "parallel_attn_mlp", False)

        block_submodules: dict[str, Any] = {
            "ln1": NormalizationBridge(
                name="input_layernorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
        }
        if not use_parallel_residual:
            block_submodules["ln2"] = NormalizationBridge(
                name="post_attention_layernorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            )
        block_submodules["attn"] = PositionEmbeddingsAttentionBridge(
            name="self_attn",
            config=self.cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
                # Per-head LN containers, present only when qk_layernorm=True
                # (stablelm-2-12b); applied post-reshape like HF.
                "q_norm": GeneralizedComponent(name="q_layernorm", optional=True),
                "k_norm": GeneralizedComponent(name="k_layernorm", optional=True),
            },
            requires_attention_mask=True,
            requires_position_embeddings=True,
        )
        block_submodules["mlp"] = self._gated_mlp()

        # StableLM has both parallel (use_parallel_residual=True) and sequential variants.
        block_cls = ParallelBlockBridge if use_parallel_residual else BlockBridge

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": block_cls(
                name="model.layers",
                submodules=block_submodules,
            ),
            "ln_final": NormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
