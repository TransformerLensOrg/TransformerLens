"""BERT architecture adapter.

This module provides the architecture adapter for BERT models.
"""

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
    PosEmbedBridge,
    UnembeddingBridge,
)


class BertArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BERT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the BERT architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # BERT uses post-LN (LayerNorm after residual, not before sublayer).
        # fold_ln assumes pre-LN (LN before sublayer) and folds ln1 into attention
        # QKV and ln2 into MLP. For post-LN, ln1 output feeds MLP (not attention)
        # and ln2 output feeds next block's attention (not MLP), so folding into
        # the wrong sublayer produces incorrect results.
        self.supports_fold_ln = False

        n_heads = self.cfg.n_heads

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_model d_head", h=n_heads
                ),
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head", h=n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (h d_head) -> h d_head d_model", h=n_heads
                ),
            ),
        }

        # Set up component mapping
        # MLM defaults; prepare_model() adjusts for other task heads (e.g., NSP).
        self.component_mapping = {
            "embed": EmbeddingBridge(name="bert.embeddings.word_embeddings"),
            "pos_embed": PosEmbedBridge(name="bert.embeddings.position_embeddings"),
            "blocks": BlockBridge(
                name="bert.encoder.layer",
                # BERT has no single MLP module (intermediate.dense and output.dense
                # are siblings in BertLayer), so the MLPBridge forward is never called
                # and mlp.hook_out never fires. Redirect hook_mlp_out to the actual
                # MLP output hook (output of the "out" linear layer).
                hook_alias_overrides={
                    "hook_mlp_out": "mlp.out.hook_out",
                    "hook_mlp_in": "mlp.in.hook_in",
                },
                submodules={
                    "ln1": NormalizationBridge(
                        name="attention.output.LayerNorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="output.LayerNorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="self.query"),
                            "k": LinearBridge(name="self.key"),
                            "v": LinearBridge(name="self.value"),
                            "o": LinearBridge(name="output.dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name=None,
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="intermediate.dense"),
                            "out": LinearBridge(name="output.dense"),
                        },
                    ),
                },
            ),
            "unembed": UnembeddingBridge(name="cls.predictions.decoder"),
            "ln_final": NormalizationBridge(
                name="cls.predictions.transform.LayerNorm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Adjust component mapping based on the actual HF model variant.

        BertForMaskedLM has cls.predictions (MLM head).
        BertForNextSentencePrediction has cls.seq_relationship (NSP head)
        and no MLM-specific LayerNorm.
        """
        if hasattr(hf_model, "cls") and hasattr(hf_model.cls, "seq_relationship"):
            # NSP model — swap head components
            assert self.component_mapping is not None
            self.component_mapping["unembed"] = UnembeddingBridge(name="cls.seq_relationship")
            self.component_mapping.pop("ln_final", None)
