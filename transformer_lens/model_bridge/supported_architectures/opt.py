"""OPT architecture adapter."""

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
    PosEmbedBridge,
    UnembeddingBridge,
)


class OptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the OPT architecture adapter."""
        super().__init__(cfg)

        # OPT models were trained with BOS tokens (inherits default_prepend_bos = True)

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "model.decoder.embed_tokens.weight",
                "pos_embed.pos": "model.decoder.embed_positions.weight",
                "blocks.{i}.ln1.w": "model.decoder.layers.{i}.self_attn_layer_norm.weight",
                "blocks.{i}.ln1.b": "model.decoder.layers.{i}.self_attn_layer_norm.bias",
                "blocks.{i}.attn.q": (
                    "model.decoder.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "model.decoder.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.v": (
                    "model.decoder.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.o": (
                    "model.decoder.layers.{i}.self_attn.out_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_Q": "model.decoder.layers.{i}.self_attn.q_proj.bias",
                "blocks.{i}.attn.b_K": "model.decoder.layers.{i}.self_attn.k_proj.bias",
                "blocks.{i}.attn.b_V": "model.decoder.layers.{i}.self_attn.v_proj.bias",
                "blocks.{i}.attn.b_O": "model.decoder.layers.{i}.self_attn.out_proj.bias",
                "blocks.{i}.ln2.w": "model.decoder.layers.{i}.final_layer_norm.weight",
                "blocks.{i}.ln2.b": "model.decoder.layers.{i}.final_layer_norm.bias",
                "blocks.{i}.mlp.in": "model.decoder.layers.{i}.fc1.weight",
                "blocks.{i}.mlp.b_in": "model.decoder.layers.{i}.fc1.bias",
                "blocks.{i}.mlp.out": "model.decoder.layers.{i}.fc2.weight",
                "blocks.{i}.mlp.b_out": "model.decoder.layers.{i}.fc2.bias",
                "ln_final.w": "model.decoder.final_layer_norm.weight",
                "ln_final.b": "model.decoder.final_layer_norm.bias",
                "unembed.u": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.decoder.embed_positions"),
            "blocks": BlockBridge(
                name="model.decoder.layers",
                submodules={
                    "ln1": NormalizationBridge(name="self_attn_layer_norm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="final_layer_norm", config=self.cfg),
                    "mlp": MLPBridge(
                        name=None,  # No MLP container; fc1/fc2 are on block
                        config=self.cfg,  # Pass config for activation function
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.decoder.final_layer_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
