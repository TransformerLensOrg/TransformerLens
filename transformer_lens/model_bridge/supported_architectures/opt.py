"""OPT architecture adapter."""

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
    NormalizationBridge,
    PosEmbedBridge,
    SymbolicBridge,
    UnembeddingBridge,
)


class OptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the OPT architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # OPT models were trained with BOS tokens (inherits default_prepend_bos = True)

        # Post-norm: disable fold_ln and center_writing_weights (pre-norm only).
        is_post_norm = not getattr(self.cfg, "do_layer_norm_before", True)
        if is_post_norm:
            self.supports_fold_ln = False
            self.supports_center_writing_weights = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        # OPT-350m is uniquely the only OPT size where word_embed_proj_dim (512)
        # != hidden_size (1024).  It uses project_in/project_out linear layers
        # instead of a final_layer_norm.  Detect this and conditionally include
        # ln_final only when the model actually has one.
        word_embed_proj_dim = getattr(self.cfg, "word_embed_proj_dim", self.cfg.d_model)
        has_final_layer_norm = word_embed_proj_dim == self.cfg.d_model

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.decoder.embed_positions"),
            "blocks": BlockBridge(
                name="model.decoder.layers",
                submodules={
                    "ln1": NormalizationBridge(
                        name="self_attn_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        requires_attention_mask=True,  # OPT requires attention_mask
                        attention_mask_4d=True,  # OPT expects 4D mask [batch, 1, tgt_len, src_len]
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(
                        name="final_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    # OPT has fc1/fc2 directly on the block, not in an MLP container.
                    # Use SymbolicBridge to maintain TransformerLens structure while
                    # correctly mapping to the underlying architecture.
                    "mlp": SymbolicBridge(
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
        if has_final_layer_norm:
            self.component_mapping["ln_final"] = NormalizationBridge(
                name="model.decoder.final_layer_norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            )
        # project_in/project_out bridge word_embed_proj_dim <-> hidden_size.
        if not has_final_layer_norm:
            self.component_mapping["project_in"] = LinearBridge(name="model.decoder.project_in")
            self.component_mapping["project_out"] = LinearBridge(name="model.decoder.project_out")
