"""NeoX architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
    SplitHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.chain_hook_conversion import (
    ChainHookConversion,
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


class NeoxArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NeoX models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the NeoX architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.cfg.default_prepend_bos = False

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "gpt_neox.embed_in.weight",
                "blocks.{i}.ln1.w": "gpt_neox.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "gpt_neox.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "gpt_neox.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "gpt_neox.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainHookConversion(
                        [
                            SplitHookConversion(0, 3),
                            RearrangeHookConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.cfg.n_heads,
                                d_head=self.cfg.d_model // self.cfg.n_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.k": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainHookConversion(
                        [
                            SplitHookConversion(1, 3),
                            RearrangeHookConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.cfg.n_heads,
                                d_head=self.cfg.d_model // self.cfg.n_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainHookConversion(
                        [
                            SplitHookConversion(2, 3),
                            RearrangeHookConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.cfg.n_heads,
                                d_head=self.cfg.d_model // self.cfg.n_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainHookConversion(
                        [
                            SplitHookConversion(0, 3),
                            RearrangeHookConversion(
                                "(head d_head) -> head d_head",
                                head=self.cfg.n_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_K": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainHookConversion(
                        [
                            SplitHookConversion(1, 3),
                            RearrangeHookConversion(
                                "(head d_head) -> head d_head",
                                head=self.cfg.n_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_V": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainHookConversion(
                        [
                            SplitHookConversion(2, 3),
                            RearrangeHookConversion(
                                "(head d_head) -> head d_head",
                                head=self.cfg.n_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "gpt_neox.layers.{i}.attention.dense.weight",
                    RearrangeHookConversion("d_model (head d_head) -> head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "gpt_neox.layers.{i}.attention.dense.bias",
                "blocks.{i}.mlp.in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                "blocks.{i}.mlp.b_in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                "blocks.{i}.mlp.out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                "blocks.{i}.mlp.b_out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                "ln_final.w": "gpt_neox.final_layer_norm.weight",
                "ln_final.b": "gpt_neox.final_layer_norm.bias",
                "unembed.u": "embed_out.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="gpt_neox.embed_in"),
            "rotary_emb": EmbeddingBridge(name="gpt_neox.rotary_emb"),
            "blocks": BlockBridge(
                name="gpt_neox.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm"),
                    "ln2": NormalizationBridge(name="post_attention_layernorm"),
                    "attn": JointQKVAttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "qkv": QKVBridge(
                                name="query_key_value",
                                config=self.cfg,
                            ),
                            "o": LinearBridge(name="dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="dense_h_to_4h"),
                            "out": LinearBridge(name="dense_4h_to_h"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="gpt_neox.final_layer_norm"),
            "unembed": UnembeddingBridge(name="embed_out"),
        }
