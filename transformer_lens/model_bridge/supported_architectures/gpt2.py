"""GPT2 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
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
    QKVBridge,
    UnembeddingBridge,
)


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT2 architecture adapter."""
        super().__init__(cfg)

        # Set default config for GPT2 models
        self.default_cfg = {
            "default_prepend_bos": True,  # Default for GPT-2 style models
        }

        self.conversion_rules = HookConversionSet(
            {
                "pos_embed.pos": "transformer.wpe.weight",
                "embed.e": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.k": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeHookConversion("(n h) m -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_Q": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_K": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_V": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.u": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": EmbeddingBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1"),
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "qkv": QKVBridge(name="c_attn", config=self.cfg),
                            "o": LinearBridge(name="c_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="ln_2"),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="c_fc"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
