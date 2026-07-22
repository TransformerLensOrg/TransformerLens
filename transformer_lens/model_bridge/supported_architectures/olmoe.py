"""OLMoE (Mixture of Experts) architecture adapter."""

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
    MoEBridge,
    MoERouterBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class OlmoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OLMoE (Mixture of Experts) models.

    OLMoE uses a pre-norm architecture with RMSNorm, Q/K normalization in attention,
    rotary position embeddings (RoPE), and sparse Mixture of Experts MLP. Key features:

    - Pre-norm: RMSNorm applied BEFORE attention and BEFORE MLP.
    - Q/K normalization: RMSNorm applied to queries and keys after projection.
    - Sparse MoE: 64 experts with top-8 routing (configurable).
    - Batched expert parameters: gate_up_proj [num_experts, 2*d_mlp, d_model] and
      down_proj [num_experts, d_model, d_mlp] as single tensors, not a ModuleList.
    - Optional QKV clipping (handled by HF's native attention forward).
    - No biases on any projections.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.ln1.b - RMSNorm has no bias
    - blocks.{i}.ln2.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the OLMoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults(final_rms=False)
        # Force eager attention for numerical consistency with benchmark reference
        self.cfg.attn_implementation = "eager"

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        # Component mapping — PRE-NORM architecture:
        # ln1 = input_layernorm (applied BEFORE attention)
        # ln2 = post_attention_layernorm (applied BEFORE MLP)
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # OLMoE uses batched expert parameters (gate_up_proj, down_proj
                    # as 3D tensors) rather than a ModuleList of individual experts.
                    # MoEBridge wraps the entire MLP module and delegates to HF's
                    # native forward pass.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": MoERouterBridge(name="gate"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Patch OLMoE's in-place clamp_ to avoid backward hook conflicts.

        Same issue as OLMo v1 — see OlmoArchitectureAdapter.prepare_model.
        """
        from transformer_lens.model_bridge.supported_architectures.olmo import (
            _patch_olmo_inplace_clamp,
        )

        _patch_olmo_inplace_clamp(hf_model)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model)
