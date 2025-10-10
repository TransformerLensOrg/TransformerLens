"""Mixtral architecture adapter."""

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
    UnembeddingBridge,
)


class MixtralArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Mixtral models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Mixtral architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "model.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "model.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "model.layers.{i}.self_attn.q_proj.bias",
                    RearrangeHookConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "model.layers.{i}.self_attn.k_proj.bias",
                    RearrangeHookConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "model.layers.{i}.self_attn.v_proj.bias",
                    RearrangeHookConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "model.layers.{i}.self_attn.o_proj.bias",
                "blocks.{i}.mlp.in": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.b_in": "model.layers.{i}.mlp.gate_proj.bias",
                "blocks.{i}.mlp.out": "model.layers.{i}.mlp.down_proj.weight",
                "blocks.{i}.mlp.b_out": "model.layers.{i}.mlp.down_proj.bias",
                "unembed.u": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "model.norm.weight",
                "ln_final.b": "model.norm.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": EmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": MoEBridge(
                        name="block_sparse_moe",
                        submodules={
                            "gate": LinearBridge(name="gate"),
                            "experts": BlockBridge(
                                name="experts",
                                submodules={
                                    "gate": LinearBridge(name="w1"),
                                    "in": LinearBridge(name="w3"),
                                    "out": LinearBridge(name="w2"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
