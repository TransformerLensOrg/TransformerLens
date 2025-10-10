"""Qwen2 architecture adapter."""

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


class Qwen2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen2 architecture adapter."""
        super().__init__(cfg)

        self.cfg.default_prepend_bos = False
        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True

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
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "num_key_value_heads", self.cfg.n_heads),
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "num_key_value_heads", self.cfg.n_heads),
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.mlp.in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.gate": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.out": "model.layers.{i}.mlp.down_proj.weight",
                "ln_final.w": "model.norm.weight",
                "unembed.u": "lm_head.weight",
            }
        )
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
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
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
