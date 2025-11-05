"""Llama architecture adapter."""

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
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class LlamaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Llama models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Llama architecture adapter."""
        super().__init__(cfg)
        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_model // cfg.n_heads,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        # Add GQA support for Llama 3.1, 3.2, and later models
        # Must set directly on cfg, not just in default_config
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        # Llama uses 'variance_epsilon' instead of 'eps' for RMSNorm
        self.cfg.eps_attr = "variance_epsilon"

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
                        n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
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

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
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
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
