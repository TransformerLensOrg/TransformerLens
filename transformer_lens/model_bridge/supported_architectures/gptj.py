"""GPTJ architecture adapter."""

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


class GptjArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPTJ models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPTJ architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.q": (
                    "transformer.h.{i}.attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "transformer.h.{i}.attn.k_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v": (
                    "transformer.h.{i}.attn.v_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o": (
                    "transformer.h.{i}.attn.out_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.fc_in.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.fc_in.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.fc_out.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.fc_out.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.u": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "attn": AttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="fc_in"),
                            "out": LinearBridge(name="fc_out"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
