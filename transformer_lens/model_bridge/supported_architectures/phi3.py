"""Phi-3 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
    SplitHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class Phi3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Phi-3 models."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.qkv_proj.weight",
                    SplitHookConversion(
                        0,
                        3,
                    ),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.qkv_proj.weight",
                    SplitHookConversion(
                        1,
                        3,
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.qkv_proj.weight",
                    SplitHookConversion(
                        2,
                        3,
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.mlp.in": (
                    "model.layers.{i}.mlp.gate_up_proj.weight",
                    SplitHookConversion(1, 2, dim=1),
                ),
                "blocks.{i}.mlp.gate": (
                    "model.layers.{i}.mlp.gate_up_proj.weight",
                    SplitHookConversion(0, 2, dim=1),
                ),
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
                            # Phi-3 uses combined qkv_proj, but we still need submodules for hooks
                            "q": LinearBridge(name="qkv_proj"),
                            "k": LinearBridge(name="qkv_proj"),
                            "v": LinearBridge(name="qkv_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # Phi-3 uses joint gate_up_proj, but we need submodules for hooks
                            "gate": LinearBridge(name="gate_up_proj"),
                            "in": LinearBridge(name="gate_up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
