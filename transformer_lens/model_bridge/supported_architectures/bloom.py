"""Bloom architecture adapter."""

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


class BloomArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Bloom models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Bloom architecture adapter."""
        super().__init__(cfg)

        self.cfg.default_prepend_bos = False
        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "transformer.word_embeddings.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.input_layernorm.bias",
                "blocks.{i}.attn.q": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeHookConversion(
                        "(three n h) m -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.k": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeHookConversion(
                        "(three n h) m -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeHookConversion(
                        "(three n h) m -> three n m h",
                        three=3,
                        n=self.cfg.n_heads,
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "transformer.h.{i}.self_attention.dense.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.b_Q": "transformer.h.{i}.self_attention.query_key_value.bias",
                "blocks.{i}.attn.b_K": "transformer.h.{i}.self_attention.query_key_value.bias",
                "blocks.{i}.attn.b_V": "transformer.h.{i}.self_attention.query_key_value.bias",
                "blocks.{i}.attn.b_O": "transformer.h.{i}.self_attention.dense.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.post_attention_layernorm.bias",
                "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.dense_h_to_4h.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.dense_h_to_4h.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.dense_4h_to_h.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.dense_4h_to_h.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.u": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.word_embeddings"),
            "embed_ln": NormalizationBridge(name="transformer.word_embeddings_layernorm"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm"),
                    "ln2": NormalizationBridge(name="post_attention_layernorm"),
                    "attn": JointQKVAttentionBridge(
                        name="self_attention",
                        config=self.cfg,
                        submodules={
                            "qkv": QKVBridge(name="query_key_value", config=self.cfg),
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
            "ln_final": NormalizationBridge(name="transformer.ln_f"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
