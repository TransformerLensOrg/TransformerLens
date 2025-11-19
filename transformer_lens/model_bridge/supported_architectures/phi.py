"""Phi architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
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


class PhiArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Phi models."""

    default_cfg = {"use_fast": False}

    def __init__(self, cfg: Any) -> None:
        """Initialize the Phi architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        self.cfg.default_prepend_bos = False

        self.weight_processing_conversions = {
            "embed.e": "transformer.wte.weight",
            "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
            "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
            "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
            "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.b_Q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_K": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_V": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n_head d_head) d_model -> n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_proj.weight",
            ),
            "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
            "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight",
            "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
            "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
            "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
            "ln_final.w": "transformer.ln_f.weight",
            "ln_final.b": "transformer.ln_f.bias",
            "unembed.u": "lm_head.weight",
            "unembed.b_U": "lm_head.bias",
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": EmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.final_layernorm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
