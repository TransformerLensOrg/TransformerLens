"""BART architecture adapter."""

from typing import Any

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


class BartArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BartForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the BART architecture adapter."""
        super().__init__(cfg)

        encoder_layers = getattr(self.cfg, "encoder_layers", self.cfg.n_layers)
        decoder_layers = getattr(self.cfg, "decoder_layers", self.cfg.n_layers)
        if encoder_layers != decoder_layers:
            raise ValueError(
                "BartArchitectureAdapter only supports symmetric BART configs for now: "
                f"encoder_layers={encoder_layers}, decoder_layers={decoder_layers}."
            )

        encoder_heads = getattr(self.cfg, "encoder_attention_heads", self.cfg.n_heads)
        decoder_heads = getattr(self.cfg, "decoder_attention_heads", self.cfg.n_heads)
        if encoder_heads != decoder_heads:
            raise ValueError(
                "BartArchitectureAdapter only supports symmetric BART attention heads for now: "
                f"encoder_attention_heads={encoder_heads}, decoder_attention_heads={decoder_heads}."
            )

        encoder_ffn_dim = getattr(self.cfg, "encoder_ffn_dim", self.cfg.d_mlp)
        decoder_ffn_dim = getattr(self.cfg, "decoder_ffn_dim", self.cfg.d_mlp)
        if encoder_ffn_dim != decoder_ffn_dim:
            raise ValueError(
                "BartArchitectureAdapter only supports symmetric BART FFN dims for now: "
                f"encoder_ffn_dim={encoder_ffn_dim}, decoder_ffn_dim={decoder_ffn_dim}."
            )

        self.cfg.n_layers = encoder_layers
        self.cfg.n_heads = encoder_heads
        self.cfg.d_head = self.cfg.d_model // encoder_heads
        self.cfg.d_mlp = encoder_ffn_dim
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # BART is post-LN. Fold-LN assumes pre-LN and would fold norms into the
        # wrong sublayers.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.encoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.encoder.embed_positions"),
            "embed_ln": NormalizationBridge(
                name="model.encoder.layernorm_embedding",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "encoder_blocks": BlockBridge(
                name="model.encoder.layers",
                hook_alias_overrides={
                    "hook_mlp_in": "mlp.in.hook_in",
                    "hook_mlp_out": "mlp.out.hook_out",
                },
                submodules={
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
                    "ln1": NormalizationBridge(
                        name="self_attn_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="final_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "mlp": SymbolicBridge(
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "decoder_embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "decoder_pos_embed": PosEmbedBridge(name="model.decoder.embed_positions"),
            "decoder_embed_ln": NormalizationBridge(
                name="model.decoder.layernorm_embedding",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "decoder_blocks": BlockBridge(
                name="model.decoder.layers",
                hook_alias_overrides={
                    "hook_attn_in": "self_attn.hook_attn_in",
                    "hook_attn_out": "self_attn.hook_out",
                    "hook_q_input": "self_attn.hook_q_input",
                    "hook_k_input": "self_attn.hook_k_input",
                    "hook_v_input": "self_attn.hook_v_input",
                    "hook_mlp_in": "mlp.in.hook_in",
                    "hook_mlp_out": "mlp.out.hook_out",
                },
                submodules={
                    "self_attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln1": NormalizationBridge(
                        name="self_attn_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "cross_attn": AttentionBridge(
                        name="encoder_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                        is_cross_attention=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="encoder_attn_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln3": NormalizationBridge(
                        name="final_layer_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
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
