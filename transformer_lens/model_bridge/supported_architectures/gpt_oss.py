"""GPT-OSS architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    NormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class GPTOSSArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-OSS model."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT-OSS architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        # GPT-OSS uses 'variance_epsilon' instead of 'eps' for RMSNorm
        self.cfg.eps_attr = "variance_epsilon"

        # Conversion rules for weight processing/folding
        # GPT-OSS uses MoE with batched experts, so we need special handling
        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                # Note: MLP weights for MoE models with batched experts are not directly mappable
                # The experts use batched tensors [num_experts, ...] which need special handling
                # These mappings are for the router only
                "ln_final.w": "model.norm.weight",
                "unembed.u": "lm_head.weight.T",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(
                        name="input_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=False,  # Avoid activation mismatches with RMSNorm
                    ),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        maintain_native_attention=True,  # Preserve GPT-OSS attention sinks
                    ),
                    "ln2": NormalizationBridge(
                        name="post_attention_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=False,  # Avoid activation mismatches with RMSNorm
                    ),
                    # GPT-OSS uses batched MoE experts with router scores
                    # MoEBridge handles the (hidden_states, router_scores) tuple returns
                    "mlp": MoEBridge(name="mlp", config=self.cfg),
                },
            ),
            "ln_final": NormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=False,  # Avoid activation mismatches with RMSNorm
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
