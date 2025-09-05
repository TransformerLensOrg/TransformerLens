"""Gemma3 architecture adapter."""


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
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class Gemma3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma3 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma3 architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                # Gemma3 scales embeddings by sqrt(d_model)
                "embed.e": (
                    "model.embed_tokens.weight",
                    RearrangeHookConversion(
                        "d_vocab d_model -> d_vocab d_model",
                        scale=self.cfg.d_model**0.5,
                    ),
                ),
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(
                            self.cfg,
                            "n_key_value_heads",
                            self.cfg.n_heads,
                        ),
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(
                            self.cfg,
                            "n_key_value_heads",
                            self.cfg.n_heads,
                        ),
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.mlp.in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.u": "lm_head.weight.T",  # Not shared with embedding
            }
        )

        # Set up component mapping with actual bridge instances
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": EmbeddingBridge(name="model.rotary_emb"),
            "rotary_emb_local": EmbeddingBridge(name="model.rotary_emb_local"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm"),
                    "ln1_post": NormalizationBridge(name="post_attention_layernorm"),
                    "ln2": NormalizationBridge(name="pre_feedforward_layernorm"),
                    "ln2_post": NormalizationBridge(name="post_feedforward_layernorm"),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": NormalizationBridge(name="q_norm"),
                            "k_norm": NormalizationBridge(name="k_norm"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
